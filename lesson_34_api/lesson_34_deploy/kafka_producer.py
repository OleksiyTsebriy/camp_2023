from kafka import KafkaProducer

topic_name = 'quickstart-events'
producer = KafkaProducer(bootstrap_servers=['localhost:9092', '192.168.88.92:9092'])

for i in range(10):
    producer.send(topic_name, f'some_message_bytes_{i}'.encode())

producer.flush()
# producer.close()
