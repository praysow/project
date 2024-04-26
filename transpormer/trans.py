import music21 as m21
#전처리
def extract_notes(midi_file):
    midi = m21.converter.parse(midi_file)
    notes_to_parse = midi.flat.notes
    notes = []

    for element in notes_to_parse:
        if isinstance(element, m21.note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, m21.chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

notes = extract_notes('path_to_midi_file.mid')
#데이터 인코딩
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_notes = encoder.fit_transform(notes)
#시퀀스 생성
sequence_length = 100
input_sequences = []
output_notes = []

for i in range(len(encoded_notes) - sequence_length):
    input_sequences.append(encoded_notes[i:i+sequence_length])
    output_notes.append(encoded_notes[i+sequence_length])
#모델 정의
import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        src = self.embedding(src)
        out = self.transformer_encoder(src)
        out = self.fc_out(out)
        return out

#모델 학습
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 하이퍼파라미터
vocab_size = len(set(encoded_notes))  # 인코딩된 노트의 고유한 값들의 개수
d_model = 512
nhead = 8
num_layers = 4
dim_feedforward = 1024
batch_size = 64
epochs = 50

model = MusicTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터셋 준비
def batch_data(source, target, batch_size):
    for i in range(0, len(source) - batch_size + 1, batch_size):
        batch_x = source[i:i+batch_size]
        batch_y = target[i:i+batch_size]
        yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)

# 학습 루프
model.train()
for epoch in range(epochs):
    total_loss = 0
    for input_seq, target_seq in batch_data(input_sequences, output_notes, batch_size):
        input_seq = input_seq.to('cuda' if torch.cuda.is_available() else 'cpu')
        target_seq = target_seq.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(input_sequences):.4f}')

#노래 생성
def generate_music(model, start_sequence, num_generate=500):
    model.eval()
    input_sequence = start_sequence
    generated_sequence = input_sequence.copy()

    for _ in range(num_generate):
        input_tensor = torch.tensor([input_sequence], dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            output = model(input_tensor)
        last_note = output[-1].argmax(-1).item()
        generated_sequence.append(last_note)
        input_sequence = generated_sequence[-sequence_length:]

    return generated_sequence

# 생성 시작 시퀀스
start_sequence = [encoded_notes[i] for i in range(sequence_length)]
generated_notes = generate_music(model, start_sequence)

# 인코딩된 노트를 다시 디코딩
decoded_notes = encoder.inverse_transform(generated_notes)
print(decoded_notes)
