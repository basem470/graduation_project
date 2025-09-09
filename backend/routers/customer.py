from fastapi import APIRouter, Query
from pydantic import BaseModel
from backend.db import sql_query_executor

customers_router = APIRouter(prefix="/customers", tags=["customers"])

class CustomerResponse(BaseModel):
    success: bool
    data: list | None = None
    message: str

@customers_router.get("/find_by_phone", response_model=CustomerResponse)
def find_customer_by_phone(customer_phone: str = Query(..., description="Customer phone number")):
    if not isinstance(customer_phone, str) or not customer_phone.strip():
        return CustomerResponse(success=False, message="Error: customer_phone must be a non-empty string.", data=None)

    sql_query = f"SELECT * FROM customers WHERE phone LIKE '%{customer_phone}%' OR phone = '{customer_phone}'"

    try:
        answer = sql_query_executor(sql_query)
        if not answer:
            return CustomerResponse(success=False, message=f"No customer found with the phone number '{customer_phone}'.", data=None)
        return CustomerResponse(success=True, message="Customer(s) found.", data=answer)
    except Exception as e:
        return CustomerResponse(success=False, message=f"Error finding customer: {e}", data=None)
