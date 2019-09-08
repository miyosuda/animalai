# 起動手順

## 初回セットアップ

```
$ sudo apt-get install docker-compose
$ git clone https://github.com/miyosuda/animalai.git
$ cd animalai
$ docker-compose build
```
でDocker Imageを作成

## 2回目以降

```
$ docker-compose up -d
$ docker exec -it animalai_lab /bin/bash
```

で起動. 作業が終わったら


```
$ docker-compose down
```

で終了






