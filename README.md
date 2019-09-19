# 起動手順

## 初回セットアップ

レポジトリをclonseする.

```
$ git clone https://github.com/miyosuda/animalai.git
```

```
$ cd animalai
$ ./scripts/setup.sh
$ ./scripts/build.sh
```
でDocker Imageを作成.


学習用にコードを変更する.

```
$ screen
````

でscreen起動

```
$ sudo docker-compose up -d
$ ./scripts/exec.sh
```

で起動.


```
$ sudo docker-compose down
```

で終了
