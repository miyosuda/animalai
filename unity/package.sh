#!/bin/sh

# MacOSX Custom
cp -rf ./CustomAnimalAI-Environment/Builds/AnimalAICustom.app .
zip -r env_mac_v1.0.0_custom.zip AnimalAICustom.app
rm -rf ./CustomAnimalAI-Environment/Builds/AnimalAICustom.app

# Linux Custom
mkdir ./env_linux_v1.0.0_custom
cp -rf ./CustomAnimalAI-Environment/Builds/AnimalAICustom_Data ./env_linux_v1.0.0_custom
cp -rf ./CustomAnimalAI-Environment/Builds/AnimalAICustom.x86_64 ./env_linux_v1.0.0_custom
zip -r env_linux_v1.0.0_custom.zip ./env_linux_v1.0.0_custom
rm -rf ./env_linux_v1.0.0_custom







