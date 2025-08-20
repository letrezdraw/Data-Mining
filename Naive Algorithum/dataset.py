weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Overcast','Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool','Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot']
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No','Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
def predict(weather_value, temp_value):
    unique_classes = set(Play)
    probabilities = {}

    for cls in unique_classes:
        prior = Play.count(cls) / len(Play)
        weather_likelihood = sum
        (
            1 for i in range(len(Play)) if weather[i] == weather_value and Play[i] == cls
        ) / Play.count(cls)
        temp_likelihood = sum
        (
            1 for i in range(len(Play)) if temp[i] == temp_value and Play[i] == cls
        ) / Play.count(cls)
        probabilities[cls] = prior * weather_likelihood * temp_likelihood
    return max(probabilities, key=probabilities.get)

result = predict('Overcast', 'Mild')
print("Prediction for ('Overcast', 'Mild'): ", result)