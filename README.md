# 起動手順

## 初回セットアップ

```
$ sudo apt-get install docker-compose
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


またGPU環境では、`docker-compose.yml` の

```
#runtime: nvidia
```

のコメントアウト部分を外す.

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

で起動. 作業が終わったら


```
$ sudo docker-compose down
```

で終了






