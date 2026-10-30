# 🧭 The AI Field Guide

> **A hands-on guide to how artificial intelligence works—and where you’ll find it in everyday life.**

AI can recommend your next song, help spot spam, forecast demand, or draft a first version of a document. This guide explores the different tools behind those experiences, with clear explanations and small Python examples you can try yourself.

> 🧭 **Keep exploring:** Continue to the [AI Agent Roadmap](https://github.com/ursleeplessprogrammer/AI-Agent-Roadmap) to learn how AI agents work and how to build one.

---

## 🗺️ The map: types of AI

### ✨ Generative AI — the creator

Generative AI makes new content by learning patterns from examples. It can produce text, images, music, video, and code.

**In the wild:** drafting an email, generating an illustration, summarizing a document, or helping write a first version of a program.

It’s a powerful starting point for creative work, but its answers can be wrong or made up. Treat generated content as a draft to review—not a source of truth.

### 🔮 Predictive AI — the forecaster

Predictive AI uses existing data to estimate what might happen next or how likely something is.

**In the wild:** forecasting product demand, estimating delivery times, or flagging a transaction that may be fraudulent.

A prediction is an estimate, not a promise. Its reliability depends on the data and assumptions behind it.

### 🏷️ Classification AI — the sorter

Classification AI assigns information to categories.

**In the wild:** filtering spam, identifying objects in photos, or routing a support request to the right team.

### 🎯 Recommendation AI — the guide

Recommendation systems suggest things a person might want to watch, read, buy, or listen to.

**In the wild:** movie suggestions, music playlists, and recommended products.

Recommendations can be useful, but they may also limit what people discover if the system keeps showing similar choices.

### 💬 Conversational AI — the conversation partner

Conversational AI lets people interact with software using everyday language, through text or speech.

**In the wild:** chatbots and voice assistants. Some follow fixed rules; others use generative AI to form more flexible responses. A natural-sounding answer isn’t necessarily a correct one.

### 👁️ Computer vision — the observer

Computer vision helps computers interpret images and video.

**In the wild:** recognizing objects in a photo, inspecting products on a production line, or organizing a large image collection.

### 🤖 Robotics and autonomous systems — the explorer

Robotics combines software with sensors and physical machines. AI can help a robot interpret its surroundings, plan a route, or respond to changing conditions.

**In the wild:** warehouse robots, robotic arms, and driver-assistance features.

---

## 🧩 Where the paths cross

These categories describe what an AI system does; they aren’t always separate technologies. One application might use computer vision to read a document, classification to identify its type, and generative AI to summarize it.

---

## 🧪 Field experiments: simple implementations

These small Python examples demonstrate the basic ideas. The data is for learning, not real-world decision-making.

### 🔮 Predictive AI: estimate a house price

The model learns from example house sizes, bedroom counts, and prices, then estimates a price for a new example.

Install the dependency:

```bash
pip install scikit-learn
```

Save as `predict_price.py`:

```python
from sklearn.linear_model import LinearRegression

# Each row contains: [size in square feet, number of bedrooms]
features = [
    [800, 1],
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
]

# Example prices in thousands of dollars
prices = [180, 220, 260, 320, 370]

model = LinearRegression()
model.fit(features, prices)

new_home = [[1300, 2]]
estimate = model.predict(new_home)[0]

print(f"Estimated price: ${estimate:.0f},000")
```

The training data is invented for this demo, so don’t use the estimate to make a real property decision.

### 🏷️ Classification AI: sort messages

This model learns to label a message as either `delivery` or `positive`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

messages = [
    "The package arrived late",
    "My delivery is delayed",
    "The product works perfectly",
    "I am happy with my purchase",
]
labels = ["delivery", "delivery", "positive", "positive"]

# Turn text into numerical features, then train a classifier.
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)
model.fit(messages, labels)

new_message = ["My order has not arrived yet"]
print(model.predict(new_message)[0])
```

This is a tiny demonstration. A useful classifier needs enough representative examples to learn from.

### ✨ Generative AI: create text

A generative AI application sends a prompt to a trained language model and uses the model’s response.

```python
# Pseudocode: use the SDK for your chosen AI provider.

prompt = "Explain predictive AI in one sentence."
response = client.generate_text(prompt)

print(response)
```

The exact code depends on the model provider. Keep API keys in environment variables or a secrets manager—never commit them to a public repository. Check generated answers before relying on them.

---

## 🧭 Start your exploration

1. Install Python and scikit-learn: `pip install scikit-learn`
2. Save one of the examples to a `.py` file.
3. Run it with `python filename.py`.
4. Change the example data and see how the output changes.

---

## ⚠️ Notes from the trail

AI systems learn from data, and data can be incomplete, outdated, or biased. Models can make mistakes and may behave poorly in situations they haven’t encountered before.

Before relying on an AI system, ask:

- What data was it trained or tested on?
- How well does it work for this specific task?
- Can a person review or challenge its output?
- Is sensitive information being handled safely?

AI is a useful tool when it has a clear purpose and responsible human oversight.

---

*This field guide is an introduction, not a complete technical manual. The best AI approach depends on the problem you’re trying to solve.*
