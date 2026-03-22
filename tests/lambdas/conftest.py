from unittest.mock import MagicMock, patch

patch("boto3.client", return_value=MagicMock()).start()
