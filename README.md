# 起動手順

## 初回セットアップ

```
$ sudo apt-get install -y docker-compose
$ git clone https://github.com/miyosuda/animalai.git
```

三好の自宅Ubuntuでは、 `docker-compose.yml` の
```
version: '2.3'
```

AWSでは

```
version: '2'
```

とそれぞれ切り替える必要がある.　(TODO: docker-composeのバージョン周りの確認)


またGPU環境では、`docker-compose.yml` 

```
#runtime: nvidia
```

のコメントアウト部分を外す必要があるが、version2では、runtimeのオプションが使えないので残す.

```
$ cd animalai
$ sudo docker-compose build
```
でDocker Imageを作成

## 2回目以降

```
$ sudo docker-compose up -d
$ sudo docker exec -it animalai_lab /bin/bash
```

で起動.

docker-composeを利用しない場合は、

```
$ docker run --runtime=nvidia -v ~/animalai:/aaio -p 6006:6006 -it --rm animalai_animalai_lab bash
```

で起動.

`docker-compose up -d` で立ち上げた場合は作業が終わったら


```
$ sudo docker-compose down
```

で終了
