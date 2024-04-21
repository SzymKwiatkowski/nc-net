#!/bin/bash

pylint $(git ls-files '*.py') --rcfile pylint.rc
