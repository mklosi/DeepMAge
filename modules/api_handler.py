import logging
import time

import requests
import timeout_decorator

# Logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

global_request_timeout = 7200  # 2h


@timeout_decorator.timeout(
    global_request_timeout,
    exception_message=f"'get_api_response' func took longer than '{global_request_timeout}' seconds.",
)
def get_api_response(
    request_type,
    url,
    headers=None,
    payload=None,
    timeout=720,
    backoff=300,
) -> requests.models.Response:
    """Send a request to an API.

    This function is agnostic to what kind of request it is. It handles retries and errors.

    :param request_type: GET or POST
    :param url: Request URL
    :param headers: Request headers
    :param payload: Request payload
    :param timeout: How many seconds to wait for a response from server before raising a 'Timeout' error.
    :param backoff: How many seconds to wait after a timeout, before we try again.

    :return: requests.models.Response
    """

    response = None
    while True:
        try:
            response = requests.request(
                request_type, url, headers=headers, data=payload, timeout=timeout
            )
            response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
            break
        except requests.exceptions.Timeout:
            logger.warning(
                f"Request to URL '{url}' timed out after '{timeout}' seconds."
            )
            logger.info(f"Backing off for '{backoff}' seconds.")
            time.sleep(backoff)
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Request to URL '{url}' failed."
                f"\n\tException: {e}"
                f"\n\tError message: {e.response.content.decode('utf-8') if e.response is not None else None}"
            )
            logger.warning(f"Retrying immediately...")
            # Sleep 1sec between retries even though we say "try immediately",
            #   otherwise on continuous errors we get flooded with logs.
            time.sleep(1)
            continue

    return response
