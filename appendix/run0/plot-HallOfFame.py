import json, pylab as pl
j = [json.loads(line) for line in open("HallOfFame.json")]
fitness = [item['fitness'][0] for item in j]
test_accuracy = [item['test_accuracy'] for item in j]


pl.plot(fitness, label = "fitness")
pl.plot(test_accuracy, label = "test accuracy")
pl.legend()
pl.savefig("time-plot.png")
pl.show()

pl.scatter(fitness, test_accuracy, label = "fitness vs test accuracy")
pl.legend()
pl.savefig("scatter-plot.png")
pl.show()


