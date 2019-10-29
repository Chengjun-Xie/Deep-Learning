import pattern


def main():
    test_checker = pattern.Checker(100, 10)
    test_checker.draw()
    test_checker.show()

    test_spectrum = pattern.Spectrum(255)
    test_spectrum.draw()
    test_spectrum.show()

    test_circle = pattern.Circle(100, 20, (50, 50))
    test_circle.draw()
    test_circle.show()

if __name__ == "__main__":
    main()


