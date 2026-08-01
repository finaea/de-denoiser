import logging

import uvicorn

from nc_bench import config

if __name__ == "__main__":
    # The embedded agents worker logs through the root logger, and uvicorn does
    # not configure one — so without this, every worker-side event (registration,
    # job offers, job failures) is discarded, which is exactly the information a
    # phone call that never records needs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    uvicorn.run("nc_bench.server:app", host="0.0.0.0", port=config.PORT, log_config=None)
