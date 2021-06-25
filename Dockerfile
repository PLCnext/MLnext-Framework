# syntax=docker/dockerfile:1
FROM python:3.8

ARG ARG_HTTP_PROXY
ARG ARG_HTTPS_PROXY
ENV HTTP_PROXY=$ARG_HTTP_PROXY
ENV HTTPS_PROXY=$ARG_HTTPS_PROXY

COPY pip.conf /etc/pip.conf

RUN pip3 install pip install mlnext \
    --index-url https://pypi:ZS2HLWUqbgmjfURn6U_7@gitlab.phoenixcontact.com/api/v4/groups/309/packages/pypi/simple
