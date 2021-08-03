#!/bin/bash

# To have more than one worker, you would need to move the Universal Sentence Encoder to another service
exec uvicorn --workers 1 'ask_currency.api:app'
