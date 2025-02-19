import React, { useState, useRef } from "react";

function App() {
    // Состояния для размеров, FFT-данных, порога и информации о компрессии
    const [width, setWidth] = useState(0);
    const [height, setHeight] = useState(0);
    const [paddedWidth, setPaddedWidth] = useState(0);
    const [paddedHeight, setPaddedHeight] = useState(0);
    const [fftData, setFftData] = useState(null); // { R, G, B } – 2D массивы комплексных чисел
    const [compressedFftData, setCompressedFftData] = useState(null);
    const [threshold, setThreshold] = useState(0);
    const [compressionInfo, setCompressionInfo] = useState("");
    const [reconstructedData, setReconstructedData] = useState(null);

    // Рефы для канвасов
    const originalCanvasRef = useRef(null);
    const spectrumCanvasRef = useRef(null);
    const reconstructedCanvasRef = useRef(null);
    const wavesSumCanvasRef = useRef(null); // для анимации суммы волн

    // Функция для вычисления следующей степени двойки
    const nextPowerOfTwo = (n) => Math.pow(2, Math.ceil(Math.log2(n)));

    // --- Комплексная арифметика ---
    const complexAdd = (a, b) => ({ re: a.re + b.re, im: a.im + b.im });
    const complexSub = (a, b) => ({ re: a.re - b.re, im: a.im - b.im });
    const complexMul = (a, b) => ({
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    });

    // --- FFT для 1D (Cooley–Tukey) ---
    const fft1d = (arr) => {
        const N = arr.length;
        if (N <= 1) return arr;
        if (N % 2 !== 0) {
            throw new Error("Длина массива должна быть степенью двойки");
        }
        const even = fft1d(arr.filter((_, i) => i % 2 === 0));
        const odd = fft1d(arr.filter((_, i) => i % 2 === 1));
        let result = new Array(N);
        for (let k = 0; k < N / 2; k++) {
            const angle = (-2 * Math.PI * k) / N;
            const twiddle = { re: Math.cos(angle), im: Math.sin(angle) };
            const t = complexMul(twiddle, odd[k]);
            result[k] = complexAdd(even[k], t);
            result[k + N / 2] = complexSub(even[k], t);
        }
        return result;
    };

    const ifft1d = (arr) => {
        const N = arr.length;
        const conj = arr.map((c) => ({ re: c.re, im: -c.im }));
        const fftConj = fft1d(conj);
        return fftConj.map((c) => ({ re: c.re / N, im: -c.im / N }));
    };

    // --- 2D FFT (сначала по строкам, затем по столбцам) ---
    const fft2d = (matrix) => {
        const N = matrix.length;
        const M = matrix[0].length;
        let result = new Array(N);
        for (let i = 0; i < N; i++) {
            result[i] = fft1d(matrix[i]);
        }
        for (let j = 0; j < M; j++) {
            let col = new Array(N);
            for (let i = 0; i < N; i++) {
                col[i] = result[i][j];
            }
            col = fft1d(col);
            for (let i = 0; i < N; i++) {
                result[i][j] = col[i];
            }
        }
        return result;
    };

    // --- 2D обратный FFT ---
    const ifft2d = (matrix) => {
        const N = matrix.length;
        const M = matrix[0].length;
        let result = new Array(N);
        for (let i = 0; i < N; i++) {
            result[i] = ifft1d(matrix[i]);
        }
        for (let j = 0; j < M; j++) {
            let col = new Array(N);
            for (let i = 0; i < N; i++) {
                col[i] = result[i][j];
            }
            col = ifft1d(col);
            for (let i = 0; i < N; i++) {
                result[i][j] = col[i];
            }
        }
        return result;
    };

    // --- Отрисовка оригинального изображения ---
    const drawOriginalImage = (imgDataObj) => {
        if (!originalCanvasRef.current) return;
        originalCanvasRef.current.width = imgDataObj.width;
        originalCanvasRef.current.height = imgDataObj.height;
        const ctx = originalCanvasRef.current.getContext("2d");
        ctx.putImageData(imgDataObj, 0, 0);
    };

    // --- Отрисовка спектрального представления ---
    const drawSpectrum = (fftR, fftG, fftB, w, h) => {
        if (!spectrumCanvasRef.current) return;
        spectrumCanvasRef.current.width = w;
        spectrumCanvasRef.current.height = h;
        const ctx = spectrumCanvasRef.current.getContext("2d");
        let magMatrix = new Array(h);
        let maxMag = 0;
        for (let i = 0; i < h; i++) {
            magMatrix[i] = new Array(w);
            for (let j = 0; j < w; j++) {
                const rMag = Math.sqrt(fftR[i][j].re ** 2 + fftR[i][j].im ** 2);
                const gMag = Math.sqrt(fftG[i][j].re ** 2 + fftG[i][j].im ** 2);
                const bMag = Math.sqrt(fftB[i][j].re ** 2 + fftB[i][j].im ** 2);
                const avgMag = (rMag + gMag + bMag) / 3;
                const logMag = Math.log(1 + avgMag);
                magMatrix[i][j] = logMag;
                if (logMag > maxMag) maxMag = logMag;
            }
        }
        const imageData = ctx.createImageData(w, h);
        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++) {
                const norm = (magMatrix[i][j] / maxMag) * 255;
                const idx = (i * w + j) * 4;
                imageData.data[idx] = norm;
                imageData.data[idx + 1] = norm;
                imageData.data[idx + 2] = norm;
                imageData.data[idx + 3] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);
    };

    // --- Отрисовка восстановленного изображения (обрезается до исходного размера) ---
    const drawReconstructedImage = (rMatrix, gMatrix, bMatrix, targetCanvas) => {
        if (!targetCanvas) return;
        targetCanvas.width = width;
        targetCanvas.height = height;
        const ctx = targetCanvas.getContext("2d");
        const imgData = ctx.createImageData(width, height);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                const idx = (i * width + j) * 4;
                const r = Math.max(0, Math.min(255, Math.round(rMatrix[i][j].re)));
                const g = Math.max(0, Math.min(255, Math.round(gMatrix[i][j].re)));
                const b = Math.max(0, Math.min(255, Math.round(bMatrix[i][j].re)));
                imgData.data[idx] = r;
                imgData.data[idx + 1] = g;
                imgData.data[idx + 2] = b;
                imgData.data[idx + 3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);
    };

    // --- Обработка загрузки файла ---
    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                // Получаем исходное изображение в полном разрешении
                const canvas = document.createElement("canvas");
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0);
                const imgDataObj = ctx.getImageData(0, 0, img.width, img.height);
                setWidth(img.width);
                setHeight(img.height);
                drawOriginalImage(imgDataObj);

                // Вычисляем размеры с паддингом (следующая степень двойки)
                const pW = nextPowerOfTwo(img.width);
                const pH = nextPowerOfTwo(img.height);
                setPaddedWidth(pW);
                setPaddedHeight(pH);

                // Создаём 2D-массивы для каждого канала (R, G, B) размером pH x pW, заполняем нулями
                const createChannelMatrix = () => {
                    let mat = new Array(pH);
                    for (let i = 0; i < pH; i++) {
                        mat[i] = new Array(pW);
                        for (let j = 0; j < pW; j++) {
                            mat[i][j] = { re: 0, im: 0 };
                        }
                    }
                    return mat;
                };

                let rMatrix = createChannelMatrix();
                let gMatrix = createChannelMatrix();
                let bMatrix = createChannelMatrix();

                // Заполняем матрицы пиксельными значениями
                for (let i = 0; i < img.height; i++) {
                    for (let j = 0; j < img.width; j++) {
                        const idx = (i * img.width + j) * 4;
                        rMatrix[i][j] = { re: imgDataObj.data[idx], im: 0 };
                        gMatrix[i][j] = { re: imgDataObj.data[idx + 1], im: 0 };
                        bMatrix[i][j] = { re: imgDataObj.data[idx + 2], im: 0 };
                    }
                }

                // Вычисляем 2D-FFT для каждого канала
                const fftR = fft2d(rMatrix);
                const fftG = fft2d(gMatrix);
                const fftB = fft2d(bMatrix);
                setFftData({ R: fftR, G: fftG, B: fftB });
                setCompressedFftData({ R: fftR, G: fftG, B: fftB });

                // Отрисовываем спектральное представление с размерами паддинга
                drawSpectrum(fftR, fftG, fftB, pW, pH);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    };

    // --- Soft thresholding по классической схеме с масштабированием ---
    const applySoftThreshold = (fftChannel, thresholdPercent) => {
        const H = fftChannel.length;
        const W = fftChannel[0].length;
        let maxMag = 0;
        for (let i = 0; i < H; i++) {
            for (let j = 0; j < W; j++) {
                const mag = Math.sqrt(fftChannel[i][j].re ** 2 + fftChannel[i][j].im ** 2);
                if (mag > maxMag) maxMag = mag;
            }
        }
        // Масштабирование порога: чем больше thresholdScale, тем ниже T
        const thresholdScale = 1000;
        const T = thresholdPercent === 0 ? 0 : (thresholdPercent / 100) * (maxMag / thresholdScale);
        let newFft = new Array(H);
        let preservedCount = 0;
        for (let i = 0; i < H; i++) {
            newFft[i] = new Array(W);
            for (let j = 0; j < W; j++) {
                const coeff = fftChannel[i][j];
                const mag = Math.sqrt(coeff.re ** 2 + coeff.im ** 2);
                if (mag <= T) {
                    newFft[i][j] = { re: 0, im: 0 };
                } else {
                    const factor = (mag - T) / mag;
                    newFft[i][j] = { re: coeff.re * factor, im: coeff.im * factor };
                    preservedCount++;
                }
            }
        }
        return { newFft, effectiveCount: preservedCount, total: H * W };
    };

    // --- Обработка изменения порога компрессии ---
    const handleThresholdChange = (e) => {
        const val = parseInt(e.target.value);
        setThreshold(val);
        if (fftData) {
            const compR = applySoftThreshold(fftData.R, val);
            const compG = applySoftThreshold(fftData.G, val);
            const compB = applySoftThreshold(fftData.B, val);
            setCompressedFftData({ R: compR.newFft, G: compG.newFft, B: compB.newFft });
            const preserved = Math.round(
                (compR.effectiveCount + compG.effectiveCount + compB.effectiveCount) / 3
            );
            const total = compR.total;
            const compressionRatio = (((total - preserved) / total) * 100).toFixed(2);
            setCompressionInfo(
                `Сохранено коэффициентов (не обнулено): ${preserved} из ${total} (порог: ${val}%). ` +
                `Сжатие: ${compressionRatio}%`
            );
            drawSpectrum(compR.newFft, compG.newFft, compB.newFft, paddedWidth, paddedHeight);
        }
    };

    // --- Обратное преобразование FFT (для каждого канала) и сборка изображения ---
    const handleInverseTransform = () => {
        if (compressedFftData) {
            const ifftR = ifft2d(compressedFftData.R);
            const ifftG = ifft2d(compressedFftData.G);
            const ifftB = ifft2d(compressedFftData.B);
            let croppedR = [];
            let croppedG = [];
            let croppedB = [];
            for (let i = 0; i < height; i++) {
                croppedR.push(ifftR[i].slice(0, width));
                croppedG.push(ifftG[i].slice(0, width));
                croppedB.push(ifftB[i].slice(0, width));
            }
            setReconstructedData({ R: croppedR, G: croppedG, B: croppedB });
            drawReconstructedImage(croppedR, croppedG, croppedB, reconstructedCanvasRef.current);
        }
    };// --- Анимация восстановления в 4 этапа с заданными процентами ---
    const animateReconstruction = () => {
        if (!compressedFftData) return;
        const totalCoeffs = paddedWidth * paddedHeight;
        // Задаем этапы как доли от общего числа коэффициентов
        const steps = [0.05, 0.10, 0.20, 1.0]; // 5%, 10%, 20%, 100%
        const delay = 1000; // задержка 1000 мс (1 секунда) между этапами
        let currentStep = 0;

        const animateStep = () => {
            const percent = steps[currentStep];
            const currentIndex = Math.floor(percent * totalCoeffs);

            // Функция формирования частичного FFT для одного канала:
            const partialFftChannel = (channel) => {
                const newMatrix = [];
                let idx = 0;
                for (let i = 0; i < channel.length; i++) {
                    newMatrix[i] = [];
                    for (let j = 0; j < channel[0].length; j++) {
                        newMatrix[i][j] = idx < currentIndex ? channel[i][j] : { re: 0, im: 0 };
                        idx++;
                    }
                }
                return newMatrix;
            };

            const partialR = partialFftChannel(compressedFftData.R);
            const partialG = partialFftChannel(compressedFftData.G);
            const partialB = partialFftChannel(compressedFftData.B);

            const ifftR = ifft2d(partialR);
            const ifftG = ifft2d(partialG);
            const ifftB = ifft2d(partialB);

            let croppedR = [];
            let croppedG = [];
            let croppedB = [];
            for (let i = 0; i < height; i++) {
                croppedR.push(ifftR[i].slice(0, width));
                croppedG.push(ifftG[i].slice(0, width));
                croppedB.push(ifftB[i].slice(0, width));
            }
            // Отрисовываем промежуточное восстановленное изображение на канвасе wavesSumCanvasRef
            drawReconstructedImage(croppedR, croppedG, croppedB, wavesSumCanvasRef.current);

            currentStep++;
            if (currentStep < steps.length) {
                setTimeout(animateStep, delay);
            }
        };

        animateStep();
    };


    return (
        <div style={{ fontFamily: "sans-serif", padding: "20px" }}>
            <h1>FFT для цветного изображения без потери разрешения</h1>
            <input type="file" accept="image/*" onChange={handleFileUpload} />
            <div style={{ marginTop: "20px" }}>
                <h3>Оригинальное изображение</h3>
                <canvas ref={originalCanvasRef} style={{ border: "1px solid black" }}></canvas>
            </div>
            <div style={{ marginTop: "20px" }}>
                <h3>Спектральное изображение (с паддингом до степени двойки)</h3>
                <canvas ref={spectrumCanvasRef} style={{ border: "1px solid black" }}></canvas>
                <div>{compressionInfo}</div>
            </div>
            <div style={{ marginTop: "20px" }}>
                <label>
                    Порог компрессии (0–100%): {threshold}%
                </label>
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={threshold}
                    onChange={handleThresholdChange}
                    style={{ width: "300px", marginLeft: "10px" }}
                />
            </div>
            <div style={{ marginTop: "20px" }}>
                <button onClick={handleInverseTransform}>Обратное преобразование</button>
            </div>
            <div style={{ marginTop: "20px" }}>
                <button onClick={animateReconstruction}>
                    Анимировать восстановление (4 этапа)
                </button>
            </div>
            <div style={{ display: "flex", marginTop: "20px" }}>
                <div style={{ marginRight: "20px" }}>
                    <h3>Восстановленное изображение</h3>
                    <canvas ref={reconstructedCanvasRef} style={{ border: "1px solid black" }}></canvas>
                </div>
                <div>
                    <h3>Анимация суммы волн</h3>
                    <canvas ref={wavesSumCanvasRef} style={{ border: "1px solid black" }}></canvas>
                </div>
            </div>
        </div>
    );
}

export default App;
