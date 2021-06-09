import datetime
from typing import List, Dict

import requests
import dataclasses
import pandas as pd
from notion.client import NotionClient, CollectionRowBlock
from notion.collection import NotionDate


# ref: https://github.com/jamalex/notion-py/issues/292#issuecomment-791139647
def notion_settings() -> None:
    """Notion APIが変更されたことによる初期設定"""
    from tzlocal import get_localzone
    import notion

    def call_load_page_chunk(self, page_id):
        if self._client.in_transaction():
            self._pages_to_refresh.append(page_id)
            return

        data = {
            "pageId": page_id,
            "limit": 100,
            "cursor": {"stack": []},
            "chunkNumber": 0,
            "verticalColumns": False,
        }

        recordmap = self._client.post("loadPageChunk", data).json()["recordMap"]

        self.store_recordmap(recordmap)

    def call_query_collection(
        self,
        collection_id,
        collection_view_id,
        search="",
        type="table",
        aggregate=[],
        aggregations=[],
        filter={},
        sort=[],
        calendar_by="",
        group_by="",
    ):

        assert not (
            aggregate and aggregations
        ), "Use only one of `aggregate` or `aggregations` (old vs new format)"

        # convert singletons into lists if needed
        if isinstance(aggregate, dict):
            aggregate = [aggregate]
        if isinstance(sort, dict):
            sort = [sort]

        data = {
            "collectionId": collection_id,
            "collectionViewId": collection_view_id,
            "loader": {
                "limit": 1000000,
                "loadContentCover": True,
                "searchQuery": search,
                "userLocale": "en",
                "userTimeZone": str(get_localzone()),
                "type": type,
            },
            "query": {
                "aggregate": aggregate,
                "aggregations": aggregations,
                "filter": filter,
                "sort": sort,
            },
        }

        response = self._client.post("queryCollection", data).json()

        self.store_recordmap(response["recordMap"])

        return response["result"]

    def search_pages_with_parent(self, parent_id, search=""):
        data = {
            "query": search,
            "parentId": parent_id,
            "limit": 100,
            "spaceId": self.current_space.id,
        }
        response = self.post("searchPagesWithParent", data).json()
        self._store.store_recordmap(response["recordMap"])
        return response["results"]

    notion.store.RecordStore.call_load_page_chunk = call_load_page_chunk
    notion.store.RecordStore.call_query_collection = call_query_collection
    notion.client.NotionClient.search_pages_with_parent = search_pages_with_parent


@dataclasses.dataclass
class Notion:
    params: str

    def __post_init__(self) -> None:
        notion_settings()
        self.client = NotionClient(token_v2=self.params.token)
        self.url = self.params.url

    def get_table(self, dropna: bool = False) -> pd.DataFrame:

        table = self.client.get_collection_view(self.url)

        rows = []
        for row in table.collection.get_rows():
            rows.append(self._get_row_item(row))

        table_df = pd.DataFrame(rows, columns=list(row.get_all_properties().keys()))
        if dropna:
            table_df = table_df.dropna().reset_index(drop=True)
        return table_df

    def _get_row_item(self, row) -> List:
        items = []
        for col, item in row.get_all_properties().items():
            type_ = type(item)
            item = row.get_property(identifier=col)
            if type_ not in [list, NotionDate]:
                items.append(item)
            elif type_ == list:
                items.append(" ".join(item))
            elif type_ == NotionDate:
                items.append(item.__dict__["start"])
        return items

    def insert_rows(self, item_dict: Dict) -> None:
        table = self.client.get_collection_view(self.url)
        row = self._create_new_record(table)

        for col_name, value in item_dict.items():
            row.set_property(identifier=col_name, val=value)

    def _create_new_record(self, table):
        row_id = self.client.create_record(
            "block", parent=table.collection, type="page"
        )
        row = CollectionRowBlock(self.client, row_id)

        with self.client.as_atomic_transaction():
            for view in self.client.get_block(table.get("parent_id")).views:
                view.set("page_sort", view.get("page_sort", []) + [row_id])

        return row


@dataclasses.dataclass
class Notificator:
    run_name: str
    model_name: str
    cv: float
    process_time: float
    comment: str
    params: Dict

    def send_line(self) -> None:
        if self.params.line.token is not None:
            endpoint = "https://notify-api.line.me/api/notify"
            message = (
                f"""\n{self.run_name}\ncv: {self.cv}\ntime: {self.process_time}[min]"""
            )
            payload = {"message": message}
            headers = {"Authorization": "Bearer {}".format(self.params.line.token)}
            requests.post(endpoint, data=payload, headers=headers)

    def send_notion(self) -> None:
        notion_api = Notion(params=self.params.notion)
        notion_api.insert_rows(
            {
                "name": self.run_name,
                "created": datetime.datetime.now(),
                "model": self.model_name,
                "local_cv": self.cv,
                "time": self.process_time,
                "comment": self.comment,
            }
        )
