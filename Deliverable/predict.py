# model = load_model(filepath)

# Predict
def predict(test,model):
    prob = model.predict(np.array([test, ]))
    index = np.argmax(prob[0])
    return classes[index]

def predict(model_filepath,):
    for rounds in range(10):
        index = random.randint(0, len(x_val) - 1)
        sample_file = x_val[index]
        print("Audio:", classes[np.argmax(y_val[index])], " and predicted:", predict(sample_file))

    score_train = model.evaluate(x_tr, y_tr, verbose=0)
    score_val = model.evaluate(x_val, y_val, verbose=0)

    print('Train Score: ', score_train, '\nValidation Score: ', score_val)

    print('\nEND')