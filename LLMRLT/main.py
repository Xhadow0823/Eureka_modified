import logging
from Prompt import Prompt
from Chat import Chat
from Logger import Logger


# logging.basicConfig(level=logging.DEBUG)
# logging.debug("GG this is DEBUG")
# logging.info("GG this is INFO")
# logging.warning("GG this WARNING")
# logging.error("GG this is ERROR")

p = Prompt('FrankaLift')
# print(messages)

c = Chat()
c.set_system_content(p.initial_system)

resp = c.chat(p.initial_user)

print( resp )