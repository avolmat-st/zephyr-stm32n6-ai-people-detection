# Release Notes for zephyr-stm32n6-ai-people-detection Application

## Purpose

Zephyr Computer Vision application demonstrating the deployment of object detection models on the [STM32N6570-DK](https://docs.zephyrproject.org/latest/boards/st/stm32n6570_dk/doc/index.html).

## Features Demonstrated in This Example

- NPU accelerated quantized AI model inference under Zephyr on STM32N6570-DK

## Software components

| Name                          | Version             | Release notes
|-----                          | -------             | -------------
| STM32Cube.AI runtime          | 11.0.0              |
| lib_vision_models_pp Library  | v0.13.0             | [release notes](Lib/lib_vision_models_pp/lib_vision_models_pp/README.md)

## Update history

### V2.0.0 / February 2026

- Update STEdgeAI to version 3.0.0.
- Switch to STEdgeAI `st-ai` API.
- Switch to YoloX model
- Update zephyr-stm32-mw-isp to version 0.3.0
- Update zephyr to version 4.3.0
- Small UI changes

### V1.0.0 / October 2025

Initial Version
