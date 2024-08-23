# 従業員の職種名とミイダスの職種のマッチングを行う
従業員の職種名は自由入力（[Slack参照](https://miidas-dev.slack.com/archives/C03TEGNQPQR/p1722325808858849?thread_ts=1722309809.021989&cid=C03TEGNQPQR)）。

しかし、従業員の職種名からミイダスの職種の特定ができたら、ポジション作成などに使える。

## 実行方法
```
docker compose up -d
docker compose exec python pip install -U -qq transformers accelerate langchain langchain_community faiss-cpu FlagEmbedding sentence_transformers tqdm
docker compose exec python python main.py
```

問題なく実行できたら、`./data`ディレクトリーに`matching_result.csv`ができる。それがマッピングの結果です。

### 実行時の注意
Cloudflare Zero Trustが有効だとモデルやライブラリーのダウンロードなどができない場合が
あるため、実行してTSL/SSLエラーが発生すればCloudflare Zero Trustを無効にしてみてください。

## 今後の試み
[こちら](https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed)
に記載があるように文章の圧縮（Compression）を試みる。
