#!/usr/bin/env bash
set -e

if [ "$DEBUG" = true ] ; then
    echo 'Debugging - ON'
    nodemon --exec streamlit run app.py
else
    echo 'Debugging - OFF'
    streamlit run app.py
fi