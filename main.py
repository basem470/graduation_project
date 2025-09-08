from agents.sales_agent.salesAgent import sales_agent_exec
questions = [
    # # --- Test Cases for find_customer_by_name_tool ---
    # "which customer is named huda?",
    # "what client has the name ahmed?",
    # "Find customers with a name like 'Acme Corp'",
    
    # # --- Test Cases for find_customer_by_id_tool ---
    # "Find customer who has the id 1",
    # "Get me the customer record with ID 2",
    # "Show details for customer number 3",
    # "Which customer has ID 4?",
    # "Retrieve information about customer 5",
    # "Can you pull up the customer with id 6?",
    # "Give me the profile of customer id 7",
    # "Look up customer record with ID 8",
    # "Who is the customer that has id 9?",
    # "Fetch the data of customer 10",
    
    # # --- Test Cases for find_customer_by_email_tool ---
    # "Find the customer with the email 'fatma976@example.com'",
    # "What customer has the email 'rana652@example.com'?",
    
    # --- Test Cases for find_customer_by_phone_tool ---
    # "Who is the customer with phone number '+201955504107'?",
    # "Find the record for phone number '1753557152'",
    
    # --- Test Cases for find_customer_by_date_created_tool ---
    # "Show me the customers created last week",
    # "Find all customers added last month",
    # "Get all customers created on '2025-03-17'",
    # "List customers created between '2024-08-15' and '2024-09-15'"

    # --- Test Cases for find_customer_by_date_created_tool ---
    # "Show me the customers created last week",
    # "Find all customers added last month",
    # "Get all customers created on '2024-09-01'",
    # "List customers created between '2024-08-15' and '2024-09-15'"

    # # --- Test Cases for find_order_by_order_id_tool ---
    # "Show me the order with id 1",
    # "Find the order with id 2",
    # "What is the order with id 3?",
    # "Show me the order with id 4",

    # --- Test Cases for find_orders_by_date ---
    # "Show me the orders created last week",
    # "Find all orders added same month last year",
    # "Get all orders created on '2024-09-01'",
    # "List orders created between '2024-08-15' and '2024-09-15'",

    # "show me the orders of Sara Mostafa Inc"
    # ""

]


for q in questions:
    print("=" * 50)

    print(
        q,
    )
    print("=" * 50)
    sales_agent_exec.invoke({"input": q})
    print("=" * 50)
