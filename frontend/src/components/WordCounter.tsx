'use client';

import { useState, useEffect } from 'react';

interface WordCountResult {
  id: string;
  text: string;
  word_count: number;
  character_count: number;
  character_count_no_spaces: number;
  sentence_count: number;
  paragraph_count: number;
  created_at: string;
}

interface WordCountStats {
  word_count: number;
  character_count: number;
  character_count_no_spaces: number;
  sentence_count: number;
  paragraph_count: number;
}

export default function WordCounter() {
  const [text, setText] = useState('');
  const [stats, setStats] = useState<WordCountStats>({
    word_count: 0,
    character_count: 0,
    character_count_no_spaces: 0,
    sentence_count: 0,
    paragraph_count: 0,
  });
  const [history, setHistory] = useState<WordCountResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Calculate stats locally for real-time updates
  useEffect(() => {
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const characters = text.length;
    const charactersNoSpaces = text.replace(/\s/g, '').length;
    const sentences = text.trim() ? (text.match(/[.!?]+/g) || []).length || (text.trim() ? 1 : 0) : 0;
    const paragraphs = text.trim() ? text.split(/\n\s*\n/).filter(p => p.trim()).length : 0;

    setStats({
      word_count: words,
      character_count: characters,
      character_count_no_spaces: charactersNoSpaces,
      sentence_count: sentences,
      paragraph_count: paragraphs || (text.trim() ? 1 : 0),
    });
  }, [text]);

  // Fetch history on component mount
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/word-count/history');
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    } finally {
      setLoading(false);
    }
  };

  const saveToHistory = async () => {
    if (!text.trim()) {
      setError('Please enter some text before saving');
      return;
    }

    try {
      setSaving(true);
      setError(null);
      const response = await fetch('/api/word-count', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (response.ok) {
        const result = await response.json();
        setHistory(prev => [result, ...prev]);
        setText('');
      } else {
        setError('Failed to save. Please try again.');
      }
    } catch (err) {
      setError('Failed to save. Please check if the backend is running.');
    } finally {
      setSaving(false);
    }
  };

  const deleteFromHistory = async (id: string) => {
    try {
      const response = await fetch(`/api/word-count/${id}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setHistory(prev => prev.filter(item => item.id !== id));
      }
    } catch (err) {
      console.error('Failed to delete:', err);
    }
  };

  const loadFromHistory = (item: WordCountResult) => {
    setText(item.text);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
        Word Counter Tool
      </h1>

      {/* Text Input Area */}
      <div className="mb-6">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste your text here..."
          className="w-full h-48 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700"
        />
      </div>

      {/* Stats Display */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <div className="bg-blue-100 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-blue-600">{stats.word_count}</div>
          <div className="text-sm text-gray-600">Words</div>
        </div>
        <div className="bg-green-100 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-green-600">{stats.character_count}</div>
          <div className="text-sm text-gray-600">Characters</div>
        </div>
        <div className="bg-yellow-100 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-yellow-600">{stats.character_count_no_spaces}</div>
          <div className="text-sm text-gray-600">No Spaces</div>
        </div>
        <div className="bg-purple-100 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-purple-600">{stats.sentence_count}</div>
          <div className="text-sm text-gray-600">Sentences</div>
        </div>
        <div className="bg-pink-100 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-pink-600">{stats.paragraph_count}</div>
          <div className="text-sm text-gray-600">Paragraphs</div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-center mb-8">
        <button
          onClick={saveToHistory}
          disabled={saving || !text.trim()}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {saving ? 'Saving...' : 'Save to History'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg text-center">
          {error}
        </div>
      )}

      {/* History Section */}
      <div className="border-t pt-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">History</h2>

        {loading ? (
          <div className="text-center text-gray-500">Loading history...</div>
        ) : history.length === 0 ? (
          <div className="text-center text-gray-500">No saved entries yet</div>
        ) : (
          <div className="space-y-3">
            {history.map((item) => (
              <div
                key={item.id}
                className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1 mr-4">
                    <p className="text-gray-700 line-clamp-2 mb-2">
                      {item.text.substring(0, 150)}
                      {item.text.length > 150 && '...'}
                    </p>
                    <div className="flex flex-wrap gap-2 text-xs text-gray-500">
                      <span className="bg-gray-100 px-2 py-1 rounded">
                        {item.word_count} words
                      </span>
                      <span className="bg-gray-100 px-2 py-1 rounded">
                        {item.character_count} chars
                      </span>
                      <span className="bg-gray-100 px-2 py-1 rounded">
                        {new Date(item.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => loadFromHistory(item)}
                      className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                    >
                      Load
                    </button>
                    <button
                      onClick={() => deleteFromHistory(item.id)}
                      className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
