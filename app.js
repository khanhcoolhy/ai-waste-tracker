const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status');

let CLASSES = []; 
let session; 
let isDetecting = false; 
let lastFrameTime = 0; // Khóa FPS chống cháy máy

const offCanvas = document.createElement('canvas');
offCanvas.width = 640;
offCanvas.height = 640;
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true }); 

// --- 1. ĐỌC YAML & NẠP MODEL ---
async function loadModel() {
    try {
        statusText.innerHTML = "⏳ Đang cấu hình data.yaml...";
        statusText.style.background = "orange";

        const yamlResponse = await fetch('./data.yaml');
        if (!yamlResponse.ok) throw new Error("Lỗi YAML");
        
        const yamlText = await yamlResponse.text();
        const config = jsyaml.load(yamlText);

        if (config.names) {
            CLASSES = Array.isArray(config.names) ? config.names : Object.values(config.names);
        } else {
            throw new Error("Sai cấu trúc YAML");
        }

        statusText.innerHTML = "⏳ Đang nạp Lõi AI (Tối ưu Mobile)...";

        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        session = await ort.InferenceSession.create('./best_yolo11_best-param-tune.onnx', { 
            executionProviders: ['webgl', 'wasm'],
            graphOptimizationLevel: 'all'
        });

        statusText.innerHTML = `✅ Đang quét (${CLASSES.length} nhãn)`;
        statusText.style.background = "#4CAF50";
    } catch (error) {
        statusText.innerHTML = "❌ Lỗi hệ thống. Xem F12.";
        statusText.style.background = "red";
        console.error(error);
    }
}

// --- 2. SETUP CAMERA ---
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

// --- 3. TIỀN XỬ LÝ (CHỈNH ẢNH SANG SỐ) ---
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

// --- 4. VÒNG LẶP SUY LUẬN SIÊU TỐC ---
async function detectFrame() {
    if (!session) return;

    // Hãm phanh FPS
    const now = Date.now();
    if (now - lastFrameTime < 50) { 
        requestAnimationFrame(detectFrame);
        return;
    }
    lastFrameTime = now;
    
    // Nếu frame trước chưa xử lý xong, bỏ qua frame này
    if (isDetecting) {
        requestAnimationFrame(detectFrame);
        return;
    }
    isDetecting = true; 

    try {
        const inputTensor = preprocess(video);
        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);
        
        // Dọn sạch RAM tensor
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

            // 1. Vẽ BBox trước
            ctx.strokeStyle = boxColor; 
            ctx.lineWidth = 4; 
            const scaledX = box.x1 * scaleX;
            const scaledY = box.y1 * scaleY;
            ctx.strokeRect(scaledX, scaledY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);

            // 2. Cấu hình nhãn (Label)
            ctx.font = "bold 20px Arial"; 
            const labelText = `${CLASSES[box.classId]} ${(box.prob * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(labelText).width;
            const labelRectHeight = 30; // Chiều cao mặc định của nhãn
            const labelRectWidth = textWidth + 16; // Chiều rộng mặc định của nhãn (+16 padding)
            const labelPadX = 8; // Padding ngang trong nhãn

            // --- 👉 THUẬT TOÁN CHỐNG TRÀN NHÃN (Bounds Checking) ---
            
            // Khởi tạo tọa độ nhãn mặc định (nằm trên hộp BBox)
            let finalLabelX = scaledX;
            let finalLabelY = scaledY - labelRectHeight;

            // Kiểm tra tràn biên trên: Nếu nhãn bị đẩy ra ngoài mép trên Canvas
            if (finalLabelY < 0) {
                // Nhảy nhãn vào nằm BÊN TRONG hộp, sát mép trên
                finalLabelY = scaledY; 
            }
            
            // Kiểm tra tràn biên phải: Nếu nhãn bị đẩy ra ngoài mép phải Canvas
            if (finalLabelX + labelRectWidth > canvas.width) {
                // Căn lề phải của nhãn khớp với lề phải của Canvas
                finalLabelX = canvas.width - labelRectWidth;
            }
            // (Thêm an toàn) Kiểm tra tràn biên trái
            if (finalLabelX < 0) {
                finalLabelX = 0;
            }

            // 3. Vẽ nền nhãn khớp với tọa độ đã tối ưu
            ctx.fillStyle = boxColor;
            ctx.fillRect(finalLabelX, finalLabelY, labelRectWidth, labelRectHeight);
            
            // 4. Vẽ chữ lên nhãn
            ctx.fillStyle = "white";
            // Tọa độ chữ lùi vào padding ngang, và căn theo baseline dọc của nhãn (cách rectTop ~22px)
            const finalTextX = finalLabelX + labelPadX;
            const finalTextY = finalLabelY + 22; 
            ctx.fillText(labelText, finalTextX, finalTextY);
        });

    } catch (e) {
        console.error("Lỗi suy luận:", e);
    }

    isDetecting = false; 
    requestAnimationFrame(detectFrame); 
}

// --- 5. CHUỖI KHỞI ĐỘNG ---
async function main() {
    await loadModel();
    await setupCamera();
    
    if (video.srcObject) {
        await video.play().catch(e => console.error("Lỗi bật cam:", e));
        detectFrame(); 
    }
}

main();