# Artificial Intelligence: A Practical Guide

Artificial intelligence (AI) is a broad name for computer systems that perform tasks we usually associate with human intelligence—such as recognizing patterns, understanding language, making predictions, or generating new content.

AI isn’t one single technology. Different kinds of AI are built for different jobs, and many real-world systems combine more than one approach.

## Main categories of AI

### Generative AI

Generative AI creates new content based on patterns learned from examples. It can produce text, images, music, video, and code.

**Examples:** drafting an email, generating an illustration, summarizing a document, or helping write a first version of a program.

Generative AI can be useful for brainstorming and speeding up routine work, but it can also produce inaccurate or made-up information. Its output should be reviewed, especially when accuracy matters.

### Predictive AI

Predictive AI uses existing data to estimate what might happen next or how likely something is.

**Examples:** forecasting product demand, estimating delivery times, or flagging a transaction that may be fraudulent.

A prediction is an estimate, not a guarantee. Its quality depends on the data and assumptions behind the model.

### Classification AI

Classification AI sorts information into categories.

**Examples:** filtering spam, identifying objects in photos, or routing a support request to the right team.

Some systems assign a single label; others estimate the likelihood of several possible labels.

### Recommendation AI

Recommendation systems suggest items a person might want to watch, read, buy, or listen to. They often use a mix of past activity, item characteristics, and patterns from other users.

**Examples:** movie recommendations, music playlists, and suggested products.

Recommendations can be convenient, but they may also narrow what people see if they repeatedly prioritize similar content.

### Conversational AI

Conversational AI lets people interact with software using everyday language, through text or speech. Chatbots and voice assistants are familiar examples.

Some conversational systems follow prewritten rules; others use generative AI to respond more flexibly. In either case, a natural-sounding answer isn’t necessarily a correct one.

### Computer vision

Computer vision helps computers interpret images and video.

**Examples:** recognizing objects in a photo, inspecting products on a production line, or helping organize a large image collection.

### Robotics and autonomous systems

Robotics combines software with sensors and physical machines. AI may help a robot interpret its surroundings, plan a route, or adjust to changing conditions.

**Examples:** warehouse robots, robotic arms, and driver-assistance features.

## How these categories overlap

These labels describe *what a system does*, not always how it is built. A single application might use computer vision to read a document, classification to identify its type, and generative AI to summarize it.

## A few simple implementations

These examples are intentionally small: they show the basic idea without hiding it behind a large framework.

### Predictive AI: estimate a number

This example trains a model to estimate a house price from its size and number of bedrooms. The data is made up for demonstration, so the predictions aren’t meaningful outside this example.

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

The model learns a relationship between the example inputs and prices, then uses it to make an estimate for a new home.

### Classification AI: sort text into categories

This example uses a small set of labeled messages to classify new text. Real applications need a larger, representative dataset.

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

# Convert text into numerical features, then train a classifier.
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)
model.fit(messages, labels)

new_message = ["My order has not arrived yet"]
print(model.predict(new_message)[0])
```

### Generative AI: create text with a language model

Generative AI usually relies on a trained model rather than a few hand-written rules. The example below shows the general idea; the exact code depends on the provider and model you choose.

```python
# Pseudocode: replace `client` and the method with your provider's SDK.

prompt = "Explain predictive AI in one sentence."
response = client.generate_text(prompt)

print(response)
```

Keep API keys in environment variables or a secrets manager—don’t commit them to a public GitHub repository. Review generated output before relying on it; language models can produce convincing but incorrect answers.

## Running the Python examples

1. Install Python and scikit-learn: `pip install scikit-learn`
2. Save an example as a `.py` file.
3. Run it with `python filename.py`.

## Things to keep in mind

AI systems learn from data, and data can be incomplete, outdated, or biased. They can make mistakes and may handle unfamiliar situations poorly. For important decisions, it’s worth asking:

- What information was the system trained or tested on?
- How accurate is it for this particular task?
- Can someone review or challenge its output?
- Is sensitive information being handled appropriately?

AI works best as a tool with a clear purpose—not as a substitute for judgment in every situation.

---

*This page is an introduction, not a complete technical guide. The field is broad, and the best approach depends on the problem being solved.*
