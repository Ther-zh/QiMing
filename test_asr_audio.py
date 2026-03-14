import numpy as np
import soundfile as sf
import tempfile
import subprocess
import os
from perception.asr.funasr_asr import FunASRRecognizer

# 从视频中提取音频
def extract_audio(video_path):
    try:
        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # 使用ffmpeg提取音频
        cmd = [
            'ffmpeg', '-i', video_path, '-ac', '1', '-ar', '16000', 
            '-f', 'wav', '-y', temp_audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 读取音频文件
        audio_data, sample_rate = sf.read(temp_audio_path)
        
        # 转换为float32格式
        audio_data = audio_data.astype(np.float32)
        
        print(f"从视频中提取音频数据，长度: {len(audio_data)} 样本，采样率: {sample_rate}Hz")
        
        # 清理临时文件
        os.unlink(temp_audio_path)
        
        return audio_data
    except Exception as e:
        print(f"提取音频时出错: {e}")
        return np.array([])

# 测试ASR模型
def test_asr():
    # 初始化ASR模型
    config = {
        'model_path': '/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall'
    }
    
    try:
        asr = FunASRRecognizer(config)
        print("ASR模型加载成功")
        
        # 从视频中提取音频
        audio_data = extract_audio('video/video.mp4')
        
        if len(audio_data) > 0:
            # 测试整个音频
            print(f"测试整个音频，长度: {len(audio_data)} 样本")
            wake_detected, asr_text = asr.inference(audio_data)
            print(f"ASR识别结果: {asr_text}")
            print(f"是否检测到唤醒词: {wake_detected}")
            
            # 测试音频的前10秒
            print("\n测试音频的前10秒")
            audio_data_10s = audio_data[:160000]  # 10秒 * 16000Hz
            wake_detected, asr_text = asr.inference(audio_data_10s)
            print(f"ASR识别结果: {asr_text}")
            print(f"是否检测到唤醒词: {wake_detected}")
        else:
            print("无法提取音频数据")
            
    except Exception as e:
        print(f"测试ASR时出错: {e}")

if __name__ == "__main__":
    test_asr()
