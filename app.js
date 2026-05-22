const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const staticImg = document.getElementById('static-img');
const statusText = document.getElementById('status');
const camInstruction = document.getElementById('cam-instruction');

// Biến rỗng, sẽ tự động được điền từ file data.yaml
let CLASSES = []; 
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

// --- 2. TỰ ĐỘNG ĐỌC YAML VÀ NẠP MODEL ---
async function loadModel() {
    try {
        statusText.innerHTML = "⏳ Đang đọc cấu hình data.yaml...";
        statusText.style.background = "orange";

        // Tải file data.yaml
        const yamlResponse = await fetch('./data.yaml');
        if (!yamlResponse.ok) throw new Error("Không tìm thấy file data.yaml ở cùng thư mục");
        
        const yamlText = await yamlResponse.text();
        const config = jsyaml.load(yamlText);

        // Lấy danh sách nhãn
        if (config.names) {
            CLASSES = Array.isArray(config.names) ? config.names : Object.values(config.names);
            console.log("✅ Đã load danh sách nhãn từ YAML:", CLASSES);
            camInstruction.innerHTML = `Hỗ trợ ${CLASSES.length} nhãn: <br> ${CLASSES.join(' • ')}`;
        } else {
            throw new Error("File data.yaml không có mục 'names'");
        }

        statusText.innerHTML = "⏳ Đang tải Model AI (khoảng 18MB)...";

        // Nạp Model ONNX (Dùng WASM để máy tính/điện thoại nào cũng chạy êm ru, không bị đơ)
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        session = await ort.InferenceSession.create('./best_yolo11_best-param-tune.onnx', { 
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        statusText.innerHTML = "✅ Hệ thống sẵn sàng!";
        statusText.style.background = "#4CAF50";
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi khởi tạo hệ thống (Xem F12)";
        statusText.style.background = "red";
        console.error("Lỗi:", error);
    }
}

// --- 3. KHỞI ĐỘNG CAMERA ---
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
        statusText.innerHTML = "❌ Lỗi mở Camera. Vui lòng cấp quyền!";
        statusText.style.background = "red";
    }
}

// --- 4. TIỀN XỬ LÝ ẢNH CHUNG ---
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

// --- 5. HÀM LÕI SUY LUẬN AI ---
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
    
    // Bảng màu tự động xoay vòng cho mọi số lượng nhãn
    const PALETTE = [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", 
        "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB"
    ];

    nmsBoxes.forEach(box => {
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;

        let boxColor = PALETTE[box.classId % PALETTE.length];

        ctx.strokeStyle = boxColor; 
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x1 * scaleX, box.y1 * scaleY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);

        ctx.fillStyle = boxColor;
        // Chiều rộng nhãn tự co giãn theo độ dài của tên rác
        const labelText = `${CLASSES[box.classId]} (${(box.prob * 100).toFixed(0)}%)`;
        ctx.fillRect(box.x1 * scaleX, box.y1 * scaleY - 25, ctx.measureText(labelText).width * 2.5 + 20, 25);
        ctx.fillStyle = "white";
        ctx.font = "bold 16px Arial";
        ctx.fillText(labelText, box.x1 * scaleX + 5, box.y1 * scaleY - 5);
    });
}

// --- 6. VÒNG LẶP CHO CAMERA ---
async function detectFrame() {
    if (!session || isDetecting || currentMode !== 'video') {
        if (currentMode === 'video') requestAnimationFrame(detectFrame);
        return;
    }
    
    isDetecting = true; 
    try {
        await runInferenceAndDraw(video);
    } catch (e) {
        console.error("Lỗi vẽ hình ngầm:", e);
    }
    isDetecting = false; 
    
    if (currentMode === 'video') requestAnimationFrame(detectFrame); 
}

// --- 7. XỬ LÝ ẢNH TĨNH ---
async function processImage(imgSource) {
    if (!session) return alert("Model AI chưa nạp xong, vui lòng chờ vài giây!");
    
    statusText.innerHTML = "⏳ Đang phân tích ảnh...";
    statusText.style.background = "orange";
    ctx.clearRect(0, 0, canvas.width, canvas.height); 

    staticImg.src = imgSource;
    staticImg.onload = async () => {
        try {
            await runInferenceAndDraw(staticImg);
            statusText.innerHTML = "✅ Phân tích ảnh hoàn tất!";
            statusText.style.background = "#4CAF50";
        } catch(e) {
            console.error(e);
        }
    };
    staticImg.onerror = () => {
        alert("Không thể tải ảnh. Nếu dùng Link, trang gốc có thể đang chặn quyền truy cập (CORS). Hãy tải ảnh về rồi dùng tab Tải Ảnh nhé.");
        statusText.innerHTML = "❌ Lỗi đọc file ảnh!";
        statusText.style.background = "red";
    }
}

// Lắng nghe sự kiện Upload / URL
document.getElementById('upload-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const imgUrl = URL.createObjectURL(file);
    processImage(imgUrl);
});

document.getElementById('btn-url').addEventListener('click', () => {
    const url = document.getElementById('input-url').value.trim();
    if (!url) return alert("Fen chưa dán link kìa!");
    processImage(url);
});

// --- 8. KHỞI ĐỘNG HỆ THỐNG AN TOÀN ---
async function main() {
    // Ưu tiên 1: Tải não bộ (YAML + ONNX)
    await loadModel();
    
    // Ưu tiên 2: Xin quyền Camera
    await setupCamera();
    
    // Ưu tiên 3: Nếu được cấp quyền, bật Cam và Quét
    if (video.srcObject) {
        await video.play().catch(e => console.error("Lỗi bật camera:", e));
        detectFrame(); 
    }
}

main();