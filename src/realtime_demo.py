import numpy as np
import torch
import torch.nn as nn
import time
import os

# ── EEGNet definition ──────────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, timepoints=1001,
                 F1=8, D=2, F2=16, dropout=0.5):
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        self._to_linear = self._get_flatten_size(channels, timepoints)
        self.classifier = nn.Linear(self._to_linear, num_classes)

    def _get_flatten_size(self, channels, timepoints):
        x = torch.zeros(1, 1, channels, timepoints)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ── gesture display ────────────────────────────────────────────────────
GESTURES = {
    0: ('LEFT HAND', '←  Imagining left hand movement'),
    1: ('RIGHT HAND', '→  Imagining right hand movement'),
    2: ('FEET',        '↓  Imagining both feet movement'),
    3: ('TONGUE',      '○  Imagining tongue movement'),
}

BARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

def confidence_bar(value, width=20):
    filled = int(value * width)
    bar = '█' * filled + '░' * (width - filled)
    return bar

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def display(trial_num, total, true_label, pred_label, probs, correct_count):
    clear()
    gesture_name, emoji, description = GESTURES[pred_label]
    true_name = GESTURES[true_label][0]
    correct = pred_label == true_label
    accuracy = correct_count / trial_num * 100 if trial_num > 0 else 0

    print("=" * 55)
    print("      EEGNet Motor Imagery BCI — Live Demo")
    print("=" * 55)
    print(f"  Trial: {trial_num}/{total}    "
          f"Session accuracy: {accuracy:.1f}%")
    print("-" * 55)
    print(f"  Predicted:  {emoji}  {gesture_name}")
    print(f"  True label: {'✅' if correct else '❌'}  {true_name}")
    print(f"  {description}")
    print("-" * 55)
    print("  Confidence per class:")
    for i, (name, prob) in enumerate(zip(
            ['Left hand', 'Right hand', 'Feet', 'Tongue'], probs)):
        bar = confidence_bar(prob)
        marker = ' ◄' if i == pred_label else ''
        print(f"  {name:<12} {bar} {prob*100:5.1f}%{marker}")
    print("=" * 55)
    print("  Press Ctrl+C to stop")

# ── main ───────────────────────────────────────────────────────────────
def main():
    # load model
    model = EEGNet(num_classes=4, channels=22, timepoints=1001)
    model.load_state_dict(torch.load('../data/eegnet_model.pth',
                                      map_location='cpu'))
    model.eval()

    # load test data
    data = np.load('../data/preprocessed.npz')
    X_test = data['X_test']
    y_test = data['y_test']

    total = len(X_test)
    correct_count = 0

    print("EEGNet BCI Demo starting in 2 seconds...")
    time.sleep(2)

    try:
        for i in range(total):
            # prepare input
            x = torch.FloatTensor(X_test[i]).unsqueeze(0).unsqueeze(0)

            # classify
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1).numpy()[0]
                pred = int(np.argmax(probs))

            true_label = int(y_test[i])
            if pred == true_label:
                correct_count += 1

            # display result
            display(i + 1, total, true_label, pred, probs, correct_count)

            # simulate real-time — 4 seconds per trial
            time.sleep(1.5)

    except KeyboardInterrupt:
        print(f"\n\nDemo stopped.")
        print(f"Final accuracy: {correct_count/total*100:.1f}%")

if __name__ == '__main__':
    main()
