const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const staticImg = document.getElementById('static-img');
const statusText = document.getElementById('status');

const CLASSES = ["Hazardous waste", "Organic waste", "Inorganic waste", "Recyclable waste"];

let session; 
let isDetecting = false; 
let currentMode = 'video'; 

const offCanvas = document.createElement('canvas');
offCanvas.width = 640;
offCanvas.height = 640;
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true }); 

// --- 1. CHUYỂN TAB LOGIC ---
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
        // Cập nhật UI của Tab
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

        // Xử lý hệ thống ngầm
        const targetMode = btn.dataset.tab;
        
        if (targetMode === 'cam') {
            if (currentMode === 'video') return; 
            currentMode = 'video';
            staticImg.style.display = 'none';
            video.style.display = 'block';
            ctx.clearRect(0, 0, canvas.width, canvas.height); 
            statusText.innerHTML = "✅ Đang quét Live Camera...";
            statusText.style.background = "#4CAF50";
            await video.play();
            detectFrame();
        } else {
            // Chế độ Ảnh (Upload / URL) -> Tắt Camera ngầm
            currentMode = 'image';
            video.pause();
            video.style.display = 'none';
            staticImg.style.display = 'block';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            statusText.innerHTML = "Đang chờ ảnh đầu vào...";
            statusText.style.background = "#ffb300";
            // Xóa ảnh cũ đi cho sạch
            staticImg.removeAttribute('src'); 
        }
    });
});

// --- 2. KHỞI ĐỘNG CAMERA ---
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 640, facingMode: "environment" }
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => resolve(video);
        });
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi mở Camera!";
        statusText.style.background = "red";
    }
}

// --- 3. NẠP MODEL ONNX ---
async function loadModel() {
    try {
        statusText.innerHTML = "⏳ Đang nạp Model (Ưu tiên GPU)...";
        statusText.style.background = "orange";
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        session = await ort.InferenceSession.create('./best_yolo11_best-param-tune.onnx', { 
            executionProviders: ['webgl', 'wasm'] 
        });

        statusText.innerHTML = "✅ Model đã lên đạn! Sẵn sàng quét rác.";
        statusText.style.background = "#4CAF50";
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi nạp file ONNX.";
        statusText.style.background = "red";
        console.error(error);
    }
}

// --- 4. TIỀN XỬ LÝ (Chung cho Video/Ảnh) ---
function preprocess(sourceElement) {
    offCtx.drawImage(sourceElement, 0, 0, 640, 640);
    const imgData = offCtx.getImageData(0, 0, 640, 640).data;
    const float32Data = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        float32Data[i]                 = imgData[i * 4] / 255.0;     
        float32Data[i + 640 * 640]     = imgData[i * 4 + 1] / 255.0; 
        float32Data[i + 2 * 640 * 640] = imgData[i * 4 + 2] / 255.0; 
    }
    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

function iou(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    return intersection / (area1 + area2 - intersection);
}

// --- 5. HÀM LÕI: SUY LUẬN & VẼ BBOX ---
async function runInferenceAndDraw(sourceElement) {
    const inputTensor = preprocess(sourceElement);
    const feeds = { [session.inputNames[0]]: inputTensor };
    
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data; 
    
    const boxes = [];
    const numClasses = CLASSES.length;
    const CONF_THRESHOLD = 0.3; 

    for (let index = 0; index < 8400; index++) {
        let maxClassProb = 0;
        let classId = -1;
        for (let col = 0; col < numClasses; col++) {
            const prob = output[8400 * (4 + col) + index];
            if (prob > maxClassProb) {
                maxClassProb = prob;
                classId = col;
            }
        }
        if (maxClassProb > CONF_THRESHOLD) {
            const cx = output[8400 * 0 + index];
            const cy = output[8400 * 1 + index];
            const w  = output[8400 * 2 + index];
            const h  = output[8400 * 3 + index];
            boxes.push({
                x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
                prob: maxClassProb, classId: classId
            });
        }
    }

    boxes.sort((a, b) => b.prob - a.prob);
    const nmsBoxes = [];
    while (boxes.length > 0) {
        const current = boxes.shift();
        nmsBoxes.push(current);
        for (let i = boxes.length - 1; i >= 0; i--) {
            if (iou(current, boxes[i]) > 0.45) boxes.splice(i, 1);
        }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height); 
    
    nmsBoxes.forEach(box => {
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;

        let boxColor = "#00FF00"; 
        if(box.classId === 0) boxColor = "#FF0000"; 
        if(box.classId === 1) boxColor = "#FFA500"; 
        if(box.classId === 2) boxColor = "#808080"; 
        if(box.classId === 3) boxColor = "#0000FF"; 

        ctx.strokeStyle = boxColor; 
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x1 * scaleX, box.y1 * scaleY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);

        ctx.fillStyle = boxColor;
        ctx.fillRect(box.x1 * scaleX, box.y1 * scaleY - 25, 200, 25);
        ctx.fillStyle = "white";
        ctx.font = "bold 16px Arial";
        ctx.fillText(`${CLASSES[box.classId]} (${(box.prob * 100).toFixed(0)}%)`, box.x1 * scaleX + 5, box.y1 * scaleY - 5);
    });
}

// --- 6. VÒNG LẶP LIVE CAM ---
async function detectFrame() {
    if (!session || isDetecting || currentMode !== 'video') {
        if (currentMode === 'video') requestAnimationFrame(detectFrame);
        return;
    }
    
    isDetecting = true; 
    try {
        await runInferenceAndDraw(video);
    } catch (e) {
        console.error(e);
    }
    isDetecting = false; 
    
    if (currentMode === 'video') requestAnimationFrame(detectFrame); 
}

// --- 7. XỬ LÝ ẢNH TĨNH ---
async function processImage(imgSource) {
    if (!session) return alert("Model chưa sẵn sàng, chờ xíu!");
    
    statusText.innerHTML = "⏳ Đang phân tích ảnh...";
    statusText.style.background = "orange";
    ctx.clearRect(0, 0, canvas.width, canvas.height); 

    staticImg.src = imgSource;
    staticImg.onload = async () => {
        try {
            await runInferenceAndDraw(staticImg);
            statusText.innerHTML = "✅ Phân tích xong!";
            statusText.style.background = "#4CAF50";
        } catch(e) {
            console.error(e);
        }
    };
    staticImg.onerror = () => {
        alert("Lỗi tải ảnh! Hãy tải ảnh về máy rồi dùng nút 'Tải Ảnh' nhé.");
        statusText.innerHTML = "❌ Lỗi ảnh!";
        statusText.style.background = "red";
    }
}

// --- Lắng nghe sự kiện Upload / URL ---
document.getElementById('upload-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const imgUrl = URL.createObjectURL(file);
    processImage(imgUrl);
});

document.getElementById('btn-url').addEventListener('click', () => {
    const url = document.getElementById('input-url').value.trim();
    if (!url) return alert("Fen dán link ảnh vào trước đã!");
    processImage(url);
});

// --- KHỞI CHẠY ---
async function main() {
    await setupCamera();
    video.play();
    await loadModel();
    detectFrame(); 
}

main();