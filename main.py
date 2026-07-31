import uvicorn

from nc_bench import config

if __name__ == "__main__":
    uvicorn.run("nc_bench.server:app", host="0.0.0.0", port=config.PORT)
