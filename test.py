import streamlit as st
import joblib
# create a text input box with custom height and width



st.write("<h1 style='font-size: 48px;'>Fake News Detection</h1>", unsafe_allow_html=True)

    
user_input = st.text_area("Enter your message", height=500)


# Create a set of radio buttons
option = st.radio(
    "Choose an option",
    ("Option 1", "Option 2", "Option 3")
)

    
def manual_testing_using_RFC(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)

    return output_lable(pred_RFC[0])

def manual_testing_using_LR(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)

    return output_lable(pred_LR[0])

def manual_testing_using_DT(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_DT = DT.predict(new_xv_test)

    return output_lable(pred_DT[0])


# Show a message based on the selected option
model_method = 1;
if option == "Option 1":
    model_method = 1;
elif option == "Option 2":
    model_method = 2;
else:
    model_method = 3;

# prediction = model.manual_testing(user_input)
# create a submit button
if st.button("Submit"):
    # do something with the user input
    st.write("You entered:",model_method)
