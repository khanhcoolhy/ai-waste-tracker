const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status');

// 4 nhãn chuẩn theo biểu đồ của bro
const CLASSES = ["Hazardous waste", "Organic waste", "Inorganic waste", "Recyclable waste"];

let session; 
let isDetecting = false; // Cờ khóa an toàn chống sập ngầm

// TỐI ƯU HÓA: Tạo Canvas ẩn 1 lần duy nhất để đọc pixel (Chống tràn RAM)
const offCanvas = document.createElement('canvas');
offCanvas.width = 640;
offCanvas.height = 640;
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true }); 

// --- 1. KHỞI ĐỘNG CAMERA ---
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

// --- 2. NẠP MODEL ONNX (ÉP XUNG GPU) ---
async function loadModel() {
    try {
        statusText.innerHTML = "⏳ Đang nạp Model (Ưu tiên dùng GPU)...";
        statusText.style.background = "orange";
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        // 👉 Đã nhét 'webgl' vào trước để ép dùng Card đồ họa chạy cho mượt
        session = await ort.InferenceSession.create('./best_yolo11_best-param-tune.onnx', { 
            executionProviders: ['webgl', 'wasm'] 
        });

        statusText.innerHTML = "✅ Model đã lên đạn! Sẵn sàng quét rác.";
        statusText.style.background = "#4CAF50";
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi nạp file ONNX. Mở F12 xem chi tiết.";
        statusText.style.background = "red";
        console.error(error);
    }
}

// --- 3. TIỀN XỬ LÝ NHANH ---
function preprocess(videoElement) {
    offCtx.drawImage(videoElement, 0, 0, 640, 640);
    const imgData = offCtx.getImageData(0, 0, 640, 640).data;

    const float32Data = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        float32Data[i]                 = imgData[i * 4] / 255.0;     
        float32Data[i + 640 * 640]     = imgData[i * 4 + 1] / 255.0; 
        float32Data[i + 2 * 640 * 640] = imgData[i * 4 + 2] / 255.0; 
    }
    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

// --- 4. HẬU XỬ LÝ (LỌC BBOX TRÙNG - IOU) ---
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

// --- 5. VÒNG LẶP LIVE CAM CHUẨN ---
async function detectFrame() {
    if (!session || isDetecting) {
        requestAnimationFrame(detectFrame);
        return;
    }
    
    isDetecting = true; 

    try {
        const inputTensor = preprocess(video);
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;
        
        // Suy luận AI
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]].data; 
        
        const boxes = [];
        const numClasses = CLASSES.length;
        
        // 👉 Đã hạ ngưỡng nhạy bén xuống 15% (0.15)
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

        // Lọc nhiễu NMS
        boxes.sort((a, b) => b.prob - a.prob);
        const nmsBoxes = [];
        while (boxes.length > 0) {
            const current = boxes.shift();
            nmsBoxes.push(current);
            for (let i = boxes.length - 1; i >= 0; i--) {
                if (iou(current, boxes[i]) > 0.45) boxes.splice(i, 1);
            }
        }

        // Xóa sạch canvas cũ
        ctx.clearRect(0, 0, canvas.width, canvas.height); 
        
        // Vẽ BBox mới
        nmsBoxes.forEach(box => {
            const scaleX = canvas.width / 640;
            const scaleY = canvas.height / 640;

            let boxColor = "#00FF00"; 
            if(box.classId === 0) boxColor = "#FF0000"; // Hazardous - Đỏ
            if(box.classId === 1) boxColor = "#FFA500"; // Kitchen - Cam
            if(box.classId === 2) boxColor = "#808080"; // Other - Xám
            if(box.classId === 3) boxColor = "#0000FF"; // Recyclable - Xanh

            ctx.strokeStyle = boxColor; 
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x1 * scaleX, box.y1 * scaleY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);

            ctx.fillStyle = boxColor;
            ctx.fillRect(box.x1 * scaleX, box.y1 * scaleY - 25, 200, 25);
            ctx.fillStyle = "white";
            ctx.font = "bold 16px Arial";
            ctx.fillText(`${CLASSES[box.classId]} (${(box.prob * 100).toFixed(0)}%)`, box.x1 * scaleX + 5, box.y1 * scaleY - 5);
        });

    } catch (e) {
        console.error("Lỗi ngầm khi đang vẽ (Đã bọc chống sập): ", e);
    }

    isDetecting = false; // Mở khóa
    requestAnimationFrame(detectFrame); // Triệu hồi frame tiếp theo siêu tốc
}

// --- KHỞI CHẠY HỆ THỐNG ---
async function main() {
    await setupCamera();
    video.play();
    await loadModel();
    detectFrame(); 
}

main();