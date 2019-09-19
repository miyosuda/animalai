#!/bin/sh

# Prepare unity environment app
wget https://www.doc.ic.ac.uk/~bb1010/animalAI/env_linux_v1.0.0.zip
mv env_linux_v1.0.0.zip ./env/
unzip ./env/env_linux_v1.0.0.zip -d ./env/

# Prepre screen config
cp ./scripts/dot.screenrc ~/.screenrc

# Install docker-compose
sudo apt-get install docker-compose -y
