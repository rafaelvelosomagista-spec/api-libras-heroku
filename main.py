import tempfile
import requests
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="Libras Analysis API")

# Configura CORS para permitir requests do Loveble
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calculate_velocity(prev_landmarks, current_landmarks, scale_factor=1.0):
    """Calcula a velocidade dos movimentos com base nos landmarks."""
    if prev_landmarks is None or current_landmarks is None:
        return 0.0
    
    prev_array = np.array([[lm.x, lm.y, lm.z] for lm in prev_landmarks.landmark])
    current_array = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks.landmark])
    
    displacement = np.sqrt(np.sum((current_array - prev_array) ** 2, axis=1))
    velocity = np.mean(displacement) * scale_factor
    
    return velocity

def detect_gesture_phases(hand_speeds, timestamps, threshold=0.01, min_hold_frames=5):
    """Detecta fases gestuais com base na velocidade das mãos."""
    phases = []
    in_hold = False
    hold_start = 0
    
    for i in range(1, len(hand_speeds)):
        if hand_speeds[i] < threshold and not in_hold:
            # Início de uma possível fase de pico (hold)
            in_hold = True
            hold_start = i
        elif hand_speeds[i] >= threshold and in_hold:
            # Fim de uma fase de pico
            hold_duration = i - hold_start
            if hold_duration >= min_hold_frames:
                phases.append({
                    "start_time": timestamps[hold_start],
                    "end_time": timestamps[i],
                    "phase_type": "Pico (Hold)",
                    "confidence": 1.0 - (np.mean(hand_speeds[hold_start:i]) / threshold)
                })
            in_hold = False
    
    # Verifica se terminou em uma fase de pico
    if in_hold and (len(hand_speeds) - hold_start) >= min_hold_frames:
        phases.append({
            "start_time": timestamps[hold_start],
            "end_time": timestamps[-1],
            "phase_type": "Pico (Hold)",
            "confidence": 1.0 - (np.mean(hand_speeds[hold_start:]) / threshold)
        })
    
    return phases

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Endpoint para analisar um vídeo de Libras."""
    # Salva o vídeo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        content = await file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name
    
    try:
        # Processa o vídeo
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise HTTPException(status_code=400, detail="Não foi possível obter FPS do vídeo.")
        
        # Listas para armazenar dados de velocidade e tempo
        hand_speeds = []
        timestamps = []
        
        # Variáveis para rastreamento
        prev_left_hand = None
        prev_right_hand = None
        frame_count = 0
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Converte a imagem para RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Faz a detecção
                results = holistic.process(image)
                
                # Calcula a velocidade para cada mão
                left_hand_velocity = calculate_velocity(prev_left_hand, results.left_hand_landmarks, fps)
                right_hand_velocity = calculate_velocity(prev_right_hand, results.right_hand_landmarks, fps)
                
                # Velocidade média das duas mãos
                avg_velocity = (left_hand_velocity + right_hand_velocity) / 2
                hand_speeds.append(avg_velocity)
                timestamps.append(frame_count / fps)
                
                # Atualiza os landmarks anteriores
                prev_left_hand = results.left_hand_landmarks
                prev_right_hand = results.right_hand_landmarks
                frame_count += 1
        
        cap.release()
        
        # Detecta fases gestuais
        phases = detect_gesture_phases(hand_speeds, timestamps)
        
        # Formata a resposta
        resultados = []
        for phase in phases:
            resultados.append({
                "tempo_inicio": phase["start_time"],
                "tempo_fim": phase["end_time"],
                "sugestao_gloss": "",  # Não reconhece glosses automaticamente
                "sugestao_fase": phase["phase_type"],
                "confianca": phase["confidence"]
            })
        
        return {"status": "sucesso", "resultados": resultados}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o vídeo: {str(e)}")
    
    finally:
        # Remove o arquivo temporário
        os.unlink(temp_video_path)

@app.get("/")
async def root():
    return {"message": "API de análise de vídeos em Libras"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)