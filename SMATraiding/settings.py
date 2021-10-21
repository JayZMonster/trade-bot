import sqlalchemy
import logging

# LOG path
LOG_PATH = "log.json"
logging.basicConfig(filename='sma_bot.log',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
sma_logger = logging.getLogger('sma_logger')

# Binance API's
API_KEY = 'omCVqDzk4hbmz33iTe19WhyI1lE1hy4wCBuCqqI1jShakOMqOYVQjOYBCl9rYEea',
API_SECRET = 'ocuXZMhLM4iiTg4DPLuQKHVXiDTHC9GonTWr2OsgzcexHYAd9CmP85cX2hxGkNWG'

# Database engine
engine_db = sqlalchemy.create_engine('postgresql://postgres:200291vf@localhost:5432/binance')
