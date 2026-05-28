import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, ANY
import os
import json

import app

@pytest.mark.asyncio
@patch('app.asyncio.to_thread', new_callable=AsyncMock)
@patch('app.sd.InputStream')
@patch('app.wavfile.write')
@patch('app.np.concatenate')
async def test_record_audio(mock_concatenate, mock_write, mock_input_stream, mock_to_thread):
    # Mock the user pressing Enter twice
    mock_to_thread.return_value = ""
    mock_concatenate.return_value = []
    
    file_path = await app.record_audio()
    
    assert file_path == "temp.wav"
    mock_input_stream.assert_called_once()
    mock_write.assert_called_once()

@pytest.mark.asyncio
@patch('app.asyncio.to_thread', new_callable=AsyncMock)
async def test_chat_with_gemini(mock_to_thread):
    # 【核心改动 1】显式 Mock 掉 app.client，防止因为没有环境变量而抛出 ValueError
    app.client = MagicMock()
    
    mock_file = MagicMock()
    mock_file.name = "mock_file_name"
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "corrected_sentence": "I am good.",
        "grammar_mistakes": ["None"],
        "fluency_score": 9.5,
        "next_question": "What is Python?"
    })
    
    # 模拟三个链式调用的返回值
    mock_to_thread.side_effect = [
        mock_file,       # 1. client.files.upload 的返回
        mock_response,   # 2. client.models.generate_content 的返回
        None             # 3. client.files.delete 的返回
    ]
    
    # 执行被测函数
    result = await app.chat_with_gemini("dummy.wav")
    
    # 验证返回结果
    assert result["corrected_sentence"] == "I am good."
    assert result["fluency_score"] == 9.5
    assert result["next_question"] == "What is Python?"
    assert mock_to_thread.call_count == 3
    
    # 【核心改动 2】严格验证 asyncio.to_thread 调用的目标函数是否已经切换为新版 SDK
    call_args = mock_to_thread.call_args_list
    assert call_args[0][0][0] == app.client.files.upload            # 第一次应该调用上传
    assert call_args[1][0][0] == app.client.models.generate_content # 第二次应该调用生成
    assert call_args[2][0][0] == app.client.files.delete            # 第三次应该调用删除

@pytest.mark.asyncio
@patch('app.edge_tts.Communicate')
@patch('app.pygame.mixer.music')
@patch('app.pygame.mixer.init')
@patch('app.pygame.mixer.quit')
@patch('app.os.remove')
@patch('app.os.path.exists', return_value=True)
async def test_play_audio(mock_exists, mock_remove, mock_quit, mock_init, mock_music, mock_communicate):
    mock_comm_instance = AsyncMock()
    mock_communicate.return_value = mock_comm_instance
    
    # Simulate music playing once then stopping
    mock_music.get_busy.side_effect = [True, False]
    
    await app.play_audio("Hello")
    
    mock_comm_instance.save.assert_called_once_with("response.mp3")
    mock_init.assert_called_once()
    mock_music.load.assert_called_once_with("response.mp3")
    mock_music.play.assert_called_once()
    mock_quit.assert_called_once()
    mock_remove.assert_called_once_with("response.mp3")

def test_render_ui(capsys):
    data = {
        "corrected_sentence": "I am a software engineer.",
        "grammar_mistakes": ["Missing article 'a'"],
        "fluency_score": 8.0,
        "next_question": "Explain async/await."
    }
    app.render_ui(data)
    
    captured = capsys.readouterr()
    assert "Corrected Sentence" in captured.out
    assert "I am a software engineer." in captured.out
    assert "Grammar Mistakes:" in captured.out
    assert "Missing article 'a'" in captured.out
    assert "Fluency Score" in captured.out
    assert "8.0/10" in captured.out
    assert "Next Question" in captured.out
    assert "Explain async/await." in captured.out