import csv
import re
import math
from typing import List, Tuple, Dict
from collections import defaultdict

# load dataset csv
def load_data(filename: str, training_size: int, testing_size: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
  # init empty lists
  training_data = []
  testing_data = []

  with open(filename, newline='') as csvfile:
      spam_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
      next(spam_reader) # skip header row
      for index, row in enumerate(spam_reader):
          if len(row) != 2:
              continue  # Skip malformed rows
          label, message = row
          if index < training_size:
            training_data.append((label, message))
          elif index < training_size + testing_size:
              testing_data.append((label, message))
          else:
              break
  return training_data, testing_data

# function for processing text, removing punctuation, numbers, etc. and setting all letters to lowercase
def preprocess_text(text: str) -> str:
  # convert to lowercase
  text = text.lower()

  # remove punctuation and numbers
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\d+', '', text)

   # remove extra whitespace
  text = ' '.join(text.split())

  return text

def preprocess_dataset(dataset: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
  return [(label, preprocess_text(message)) for label, message in dataset]

def compute_prior_probabilities(dataset: List[Tuple[str, str]]) -> Dict[str, float]:
  # determine total amount of messages based on length of dataset
  total_messages = len(dataset)
  # init empty label counts object
  label_counts = {}
  # populate label counts object
  for label, _ in dataset:
    label_counts[label] = label_counts.get(label, 0) + 1
  # calculate prior probabilities
  prior_probabilities = {label: count / total_messages for label, count in label_counts.items()}

  return prior_probabilities


# split each word in the dataset to build vocab sets
def create_vocab(dataset: List[Tuple[str, str]]) -> set:
  vocab = set()
  for _, message in dataset:
    vocab.update(message.split())
  return vocab

# compute word counts
def compute_word_counts(dataset: List[Tuple[str, str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
  # init data structures - any new word inits with count of 0 due to defaultdict(int)
  word_counts = defaultdict(lambda:defaultdict(int))
  # this dict will track word count for each label
  label_word_counts = defaultdict(int)

  # iterate through dataset, go through each (label, message) pair in the dataset
  for label, message in dataset:
    # process each word (split each message into words)
    for word in message.split():
      # update word counts
      # increment total count for current word in the apprpriate label dict (*starts at 0 if word is new for label*)
      word_counts[label][word] += 1
      # increment total word count for current label
      label_word_counts[label] += 1

  return dict(word_counts), dict(label_word_counts)

def compute_conditional_probabilities(word_counts: Dict[str, Dict[str, int]],
                                      label_word_counts: Dict[str, int],
                                      vocabulary: set) -> Dict[str, Dict[str, float]]:

  conditional_probs = defaultdict(dict)
  vocabulary_size = len(vocabulary)

  for label in word_counts:
      for word in vocabulary:
          word_count = word_counts[label].get(word, 0)
          total_words = label_word_counts[label]

          # Using laplace smoothing
          prob = (word_count + 1) / (total_words + vocabulary_size)
          conditional_probs[label][word] = prob

  return dict(conditional_probs)


def classify_message(message: str,
                     prior_probabilities: Dict[str, float],
                     conditional_probabilities: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
  words = message.split()
  scores = {}

  for label in prior_probabilities:
      score = math.log(prior_probabilities[label])
      for word in words:
          if word in conditional_probabilities[label]:
              score += math.log(conditional_probabilities[label][word])
      scores[label] = score

  predicted_label = max(scores, key=scores.get)

  # convert log probabilities to actual probabilities
  total = sum(math.exp(score) for score in scores.values())
  probabilities = {label: math.exp(score) / total for label, score in scores.items()}

  return predicted_label, probabilities

def evaluate_classifier(dataset: List[Tuple[str, str]],
                        prior_probabilities: Dict[str, float],
                        conditional_probabilities: Dict[str, Dict[str, float]]) -> float:
  correct_predictions = 0
  total_predictions = len(dataset)

  for true_label, message in dataset:
      predicted_label, probabilities = classify_message(message, prior_probabilities, conditional_probabilities)
      if predicted_label == true_label:
          correct_predictions += 1

      print(f"\nMessage: {message}")
      print(f"True label: {true_label}")
      print(f"Predicted label: {predicted_label}")
      for label, prob in probabilities.items():
          print(f"P({label}|message) = {prob:.4f}")

  accuracy = correct_predictions / total_predictions
  return accuracy

def main():
  print("Naive Bayes Classifier")
  training_data, testing_data = load_data('SpamDetection.csv', training_size=20, testing_size=10)
  training_data = preprocess_dataset(training_data)
  testing_data = preprocess_dataset(testing_data)
  prior_probabilities = compute_prior_probabilities(training_data)
  vocabulary = create_vocab(training_data)
  word_counts, label_word_counts = compute_word_counts(training_data)
  conditional_probabilities = compute_conditional_probabilities(word_counts, label_word_counts, vocabulary)
  print("\nEvaluating the classifier on the test set:")
  accuracy = evaluate_classifier(testing_data, prior_probabilities, conditional_probabilities)

  print(f"\nTest set accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()