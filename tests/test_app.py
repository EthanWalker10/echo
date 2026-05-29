import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
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
@patch('app.read_json', new_callable=AsyncMock)
@patch('app.asyncio.to_thread', new_callable=AsyncMock)
async def test_chat_with_gemini(mock_to_thread, mock_read_json):
    mock_file = MagicMock()
    mock_file.name = "mock_file_name"
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "corrected_sentence": "I am good.",
        "grammar_mistakes": ["None"],
        "fluency_score": 9.5,
        "tech_correctness_score": 10.0,
        "tech_feedback": "Perfect.",
        "identified_weak_grammar": [],
        "identified_weak_topics": [],
        "identified_weak_tech_concepts": ["Python"],
        "identified_topics_discussed": ["General"],
        "identified_vocab_mistakes": [],
        "resolved_weak_tech_concepts": [],
        "resolved_weak_grammar": [],
        "next_question": "What is Python?"
    })
    
    mock_to_thread.side_effect = [
        mock_file,       # upload_file
        mock_response,   # generate_content
        None             # delete_file
    ]
    
    mock_read_json.return_value = {}
    
    # We must ensure app.client is not None. 
    app.client = MagicMock()
    
    result = await app.chat_with_gemini("dummy.wav")
    
    assert result["corrected_sentence"] == "I am good."
    assert result["fluency_score"] == 9.5
    assert result["tech_correctness_score"] == 10.0
    assert result["next_question"] == "What is Python?"
    assert mock_to_thread.call_count == 3

@pytest.mark.asyncio
@patch('app.read_json', new_callable=AsyncMock)
@patch('app.write_json', new_callable=AsyncMock)
async def test_update_memory_background(mock_write_json, mock_read_json):
    mock_read_json.side_effect = [
        {"fluency_average": 5.0, "total_rounds": 1},  # metrics
        {"weak_grammar": ["article misuse", "old grammar"], "weak_topics": [], "weak_tech_concepts": ["GIL", "old tech"], "topics_discussed": [], "common_vocab_mistakes": {}, "fluency_average": 5.0, "mastered_grammar": [], "mastered_tech_concepts": []}, # compressed
        [{"user_said": "Old", "assistant_asked": "Old Q"}] # session
    ]
    
    data = {
        "fluency_score": 9.0,
        "identified_weak_grammar": ["new grammar"],
        "identified_weak_topics": ["k8s"],
        "identified_weak_tech_concepts": ["new tech"],
        "identified_topics_discussed": ["threading"],
        "identified_vocab_mistakes": [{"wrong_phrase": "big traffic", "correct_phrase": "high throughput"}],
        "resolved_weak_grammar": ["article misuse"],
        "resolved_weak_tech_concepts": ["GIL"],
        "corrected_sentence": "New",
        "next_question": "New Q"
    }
    
    await app.update_memory_background(data)
    
    assert mock_write_json.call_count == 3

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
        "tech_correctness_score": 9.0,
        "tech_feedback": "Good job.",
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
    assert "Tech Correctness Score" in captured.out
    assert "9.0/10" in captured.out
    assert "Tech Feedback" in captured.out
    assert "Good job." in captured.out
    assert "Next Question" in captured.out
    assert "Explain async/await." in captured.out
