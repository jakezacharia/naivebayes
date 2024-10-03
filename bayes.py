# bayes.py
import csv
import math

# declare list objects
training_data = []
testing_data = []

# import csv file and split contents into training and testing data objects
with open('SpamDetection.csv', newline='') as csvfile:
  spam = csv.reader(csvfile, delimiter=' ', quotechar='|')
  for index, row in enumerate(spam):
    if index < 20:
      training_data.append(row)
    elif index < 30:
      testing_data.append(row)
    else:
      break

# ensure training/testing data are correct size, comment this out when done
print(f"Training Data: {len(training_data)} rows")
print(f"Testing Data: {len(testing_data)} rows")



def main():
  print("Naive Bayes Classifier Initialization")

if __name__ == "__main__":
  main()