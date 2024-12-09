# %%
import asyncio


# 非同期関数の定義
async def task1(name):
    print(f"{name}さん、こんにちは")
    await asyncio.sleep(1)  # 1秒間非同期で待機
    print("こんにちは!")
    return name


async def task2(name):
    print(f"{name}さん、こんばんは")
    await asyncio.sleep(2)  # 2秒間非同期で待機
    print("こんばんは!")
    return name


# 非同期タスクを実行し、結果を収集するメイン関数
async def main():
    # asyncio.gatherを使用してtask1とtask2を同時に実行
    results = await asyncio.gather(
        task1("太郎"),  # task1に"太郎"を引数として渡す
        task2("花子"),  # task2に"花子"を引数として渡す
    )
    print(results)  # 結果をリスト形式で出力


# Jupyter NotebookやIDE環境用の修正版
if __name__ == "__main__":
    try:
        # 既存のイベントループを取得
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # イベントループが存在しない場合、新規作成
        loop = None

    if loop and loop.is_running():
        # 既存のループが動作中の場合
        task = main()
        asyncio.ensure_future(task)  # タスクをスケジュール
    else:
        # 通常のイベントループを使用
        asyncio.run(main())
        asyncio.run(main())
