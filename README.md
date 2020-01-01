# Perceptron

Perceptron bir sinir ağı modelidir. Tek katmanlı ve çok katmanlı sinir ağları bulunmaktadır ve perceptron en basit tek katmanlı sinir ağıdır. Eğitilebilecek tek bir yapay sinir ağı hücresinden oluşur. Bu yüzden basit problemler için geçerlilik sağlarken, problem geliştikçe ve zorlaştıkça, bu tek katmanlı perceptron ihtiyacımızı karşılamaz.

Perceptron basitce şu şekilde uygulanabilir;

class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                

Bu python kodunda bir Perceptron class'ı görüyoruz. Perceptronda öğrenme oranı(learning_rate) genellike 0.1 ile 0.5 arasında değişir.
Ağırlık değerleri(weights) başlangıçta küçük rastgele değerler olarak belirlenir. Yapay sinir ağımıza girdiler(inputs) verildikten sonra net girdi(input) ve çıktı(output) hesaplanır. Bu çıktı(output) beklenen çıktıya eşitse ağırlıklarda(weights) herhangi bir değişme olmaz. Fakat farklı olursa perceptrona etki eden tüm ağırlıklar değiştirilir. Eğitim adım sayısı, öğrenme oranı(learning_rate) ve giriş dizisine(inputs) bağlıdır. Sonu gelen adımlar sonunda yapay sinir ağımız perceptronun eğitimi tamamlanmış olur.

OĞUZHAN TÖRE ÖZYÜREK
140 401 012
GÖRÜNTÜ İŞLEME FİNAL BLOG YAZISI
