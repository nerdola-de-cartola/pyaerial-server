from fastapi import FastAPI
from pydantic import BaseModel
from ldpc import ldpc_stack
from time import time

app = FastAPI()

class LdpcParams(BaseModel):
    esno_db: float # = 8.4
    num_prb: int # = 100
    num_layers: int # = 4

@app.post("/ldpc/")
async def ldpc(params: LdpcParams):
    start = time()
    ber, accuracy, *_ = ldpc_stack(params.esno_db, params.num_prb, params.num_layers)
    execution_time = (time() - start) * 1000

    return {
        "ber": ber,
        "accuracy": accuracy,
        "time (ms)": execution_time
    }

if __name__ == "__main__":
    import uvicorn

    print("Server running")

    uvicorn.run(
        "main:app",  # filename:variable
        host="0.0.0.0",
        port=8080,
        workers=8,
        reload=True
    )
