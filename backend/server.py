from fastapi import FastAPI
from backend.routers.customer import customers_router

app = FastAPI(title="ERP Customer API")

# Include routers
app.include_router(customers_router)

@app.get("/")
def root():
    return {"message": "ERP Customer API is running"}
