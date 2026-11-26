#!/usr/bin/env python3
"""Render demo.png: a terminal-style card with the real training results
and sample predictions from the spam detector."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 860
BG = (24, 26, 32)
PANEL = (31, 34, 42)
FG = (220, 224, 230)
DIM = (140, 146, 158)
GREEN = (80, 200, 120)
RED = (235, 95, 95)
CYAN = (90, 180, 235)
YELLOW = (230, 200, 110)

def font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()

f_title = font(30, True)
f_head = font(24, True)
f_txt = font(22)
f_small = font(20)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.rounded_rectangle([24, 24, W - 24, H - 24], radius=18, fill=PANEL, outline=(60, 64, 76), width=2)
# window dots
for i, col in enumerate([(235, 95, 95), (230, 200, 110), (80, 200, 120)]):
    d.ellipse([52 + i * 34, 48, 70 + i * 34, 66], fill=col)

y = 92
d.text((56, y), "SMS Spam Detector - Day 2", font=f_title, fill=FG); y += 52
d.text((56, y), "$ python train.py", font=f_txt, fill=CYAN); y += 40
for line, col in [
    ("Loaded 5,572 SMS messages (4,825 ham, 747 spam) - UCI SMS Spam Collection", FG),
    ("Train 4,457 / Test 1,115 - TF-IDF (words + pairs) + Multinomial Naive Bayes", FG),
    ("Test accuracy: 98.6%", GREEN),
    ("Spam: precision 100%, recall 89.3%  |  ham: precision 98.4%, recall 100%", FG),
    ("Confusion matrix: 966 ham correct, 0 ham flagged as spam, 133/149 spam caught", FG),
]:
    d.text((72, y), line, font=f_txt, fill=col); y += 38

y += 18
d.text((56, y), "$ python predict.py <message>", font=f_txt, fill=CYAN); y += 44

samples = [
    ('"Congratulations! You\'ve won a FREE iPhone 15. Click here to claim!"', "SPAM", "99.7%", RED),
    ('"URGENT: your bank account is locked. Verify at secure-login-now.com"', "SPAM", "93.9%", RED),
    ('"Hey, are we still meeting for lunch at 1pm tomorrow?"', "HAM", "100.0%", GREEN),
    ('"Can you pick up milk on the way home?"', "HAM", "100.0%", GREEN),
]
for msg, verdict, conf, col in samples:
    d.text((72, y), msg, font=f_small, fill=FG); y += 34
    d.text((92, y), f"-> {verdict} (confidence {conf})", font=f_small, fill=col); y += 46

d.text((56, H - 66), "github.com/youssefdahroug/ai-spam-detector", font=f_small, fill=DIM)

img.save("/downloads/ai-spam-detector/demo.png")
print("saved demo.png", img.size)
