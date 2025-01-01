from zyk import LM

if __name__ == "__main__":
    lm = LM(
        model_name="gemini-2-flash-exp",
        formatting_model_name="gemini-2-flash-exp",
        temperature=0.1,
    )
    response = lm.respond_sync(
        system_message="Please answer helpfully",
        user_message="Hello, how can I help you today? What is your name",
    )
    print(response)
