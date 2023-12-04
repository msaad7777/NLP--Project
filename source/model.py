from sklearn.naive_bayes import MultinomialNB

def train_classifier(X_train, y_train):
    classifier = MultinomialNB(alpha=0.1).fit(X_train, y_train)
    return classifier
