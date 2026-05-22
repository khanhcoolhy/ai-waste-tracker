const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const staticImg = document.getElementById('static-img');
const statusText = document.getElementById('status');
const camInstruction = document.getElementById('cam-instruction');

let CLASSES = []; 
let session; 
let isDetecting = false; 
let currentMode = 'video'; 
let lastFrameTime = 0; // Khóa FPS chống cháy máy

const offCanvas = document.createElement('canvas');
offCanvas.width = 640;
offCanvas.height = 640;
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true }); 

// --- 1. GIAO DIỆN CHUYỂN TAB ---
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

        const targetMode = btn.dataset.tab;
        
        if (targetMode === 'cam') {
            if (currentMode === 'video') return; 
            currentMode = 'video';
            staticImg.style.display = 'none';
            video.style.display = 'block';
            ctx.clearRect(0, 0, canvas.width, canvas.height); 
            statusText.innerHTML = "✅ Đang quét Live Camera...";
            statusText.style.background = "#4CAF50";
            if(video.srcObject) await video.play();
            detectFrame();
        } else {
            currentMode = 'image';
            video.pause();
            video.style.display = 'none';
            staticImg.style.display = 'block';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            statusText.innerHTML = "Đang chờ tải ảnh đầu vào...";
            statusText.style.background = "#ffb300";
            staticImg.removeAttribute('src'); 
        }
    });
});

// --- 2. NẠP DỮ LIỆU ĐỘNG & AI ---
async function loadModel() {
    try {
        statusText.innerHTML = "⏳ Đang đọc cấu hình data.yaml...";
        statusText.style.background = "orange";

        const yamlResponse = await fetch('./data.yaml');
        if (!yamlResponse.ok) throw new Error("Không tìm thấy data.yaml");
        
        const yamlText = await yamlResponse.text();
        const config = jsyaml.load(yamlText);

        if (config.names) {
            CLASSES = Array.isArray(config.names) ? config.names : Object.values(config.names);
            camInstruction.innerHTML = `Đang hỗ trợ ${CLASSES.length} nhãn: <br> <b>${CLASSES.join(' • ')}</b>`;
        } else {
            throw new Error("YAML thiếu mảng 'names'");
        }

        statusText.innerHTML = "⏳ Đang khởi động lõi AI (Ưu tiên GPU)...";

        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        session = await ort.InferenceSession.create('./best_yolo11_best-param-tune.onnx', { 
            executionProviders: ['webgl', 'wasm'],
            graphOptimizationLevel: 'all'
        });

        statusText.innerHTML = "✅ Hệ thống sẵn sàng hoạt động!";
        statusText.style.background = "#4CAF50";
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi hệ thống. Đang thiếu file hoặc sai đường dẫn.";
        statusText.style.background = "red";
        console.error(error);
    }
}

// --- 3. MỞ CAMERA SAU ---
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 640, facingMode: "environment" } // Bắt buộc dùng cam sau trên mobile
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => resolve(video);
        });
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi mở Camera. Chưa cấp quyền truy cập!";
        statusText.style.background = "red";
    }
}

// --- 4. TOÁN HỌC & TIỀN XỬ LÝ ---
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

// --- 5. BỘ NÃO NHẬN DIỆN & VẼ HÌNH ---
async function runInferenceAndDraw(sourceElement) {
    const inputTensor = preprocess(sourceElement);
    const feeds = { [session.inputNames[0]]: inputTensor };
    
    const results = await session.run(feeds);
    
    // 👉 TỐI ƯU CỰC MẠNH: Dọn rác RAM ngay lập tức
    inputTensor.dispose(); 
    
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
    
    const PALETTE = ["#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB"];

    nmsBoxes.forEach(box => {
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;
        let boxColor = PALETTE[box.classId % PALETTE.length];

        ctx.strokeStyle = boxColor; 
        ctx.lineWidth = 4; // 👉 Viền BBox dày hơn để dễ nhìn
        ctx.strokeRect(box.x1 * scaleX, box.y1 * scaleY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);

        ctx.fillStyle = boxColor;
        const labelText = `${CLASSES[box.classId]} ${(box.prob * 100).toFixed(0)}%`;
        
        // 👉 CHỮ TO HƠN, KHUNG RỘNG HƠN DỄ ĐỌC TRÊN MOBILE
        ctx.font = "bold 20px Arial"; 
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(box.x1 * scaleX, box.y1 * scaleY - 30, textWidth + 16, 30);
        
        ctx.fillStyle = "white";
        ctx.fillText(labelText, box.x1 * scaleX + 8, box.y1 * scaleY - 8);
    });
}

// --- 6. VÒNG LẶP CHỐNG LAG ---
async function detectFrame() {
    if (!session || isDetecting || currentMode !== 'video') {
        if (currentMode === 'video') requestAnimationFrame(detectFrame);
        return;
    }

    // 👉 HÃM PHANH FPS: Quét 20 lần/giây để giải phóng CPU điện thoại
    const now = Date.now();
    if (now - lastFrameTime < 50) { 
        requestAnimationFrame(detectFrame);
        return;
    }
    lastFrameTime = now;
    
    isDetecting = true; 
    try {
        await runInferenceAndDraw(video);
    } catch (e) {
        console.error("Lỗi vẽ ngầm:", e);
    }
    isDetecting = false; 
    
    if (currentMode === 'video') requestAnimationFrame(detectFrame); 
}

// --- 7. TÍNH NĂNG ẢNH ---
async function processImage(imgSource) {
    if (!session) return alert("Hệ thống AI chưa sẵn sàng!");
    
    statusText.innerHTML = "⏳ Đang phân tích bằng AI...";
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
        alert("Lỗi tải ảnh! Hãy thử lưu ảnh về máy rồi dùng nút 'Tải Ảnh' nhé.");
        statusText.innerHTML = "❌ Lỗi file ảnh!";
        statusText.style.background = "red";
    }
}

document.getElementById('upload-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const imgUrl = URL.createObjectURL(file);
    processImage(imgUrl);
});

document.getElementById('btn-url').addEventListener('click', () => {
    const url = document.getElementById('input-url').value.trim();
    if (!url) return;
    processImage(url);
});

// --- 8. CHUỖI KHỞI ĐỘNG ---
async function main() {
    await loadModel();
    await setupCamera();
    
    if (video.srcObject) {
        await video.play().catch(e => console.error("Lỗi bật cam:", e));
        detectFrame(); 
    }
}

main();