import pika 

# Connect to RabbitMQ server (running on localhost)
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Make sure a queue named 'hello' exists
channel.queue_declare(queue='hello')

# Send a message to that queue
channel.basic_publish(exchange='', routing_key='hello', body='Hello Kazi!')
print(" [x] Sent 'Hello Kazi!'")

# Close the connection
connection.close()
