from graph.graph_flow import app

user_input = input("Enter your question\n")

print(app.invoke({"question": user_input}))
