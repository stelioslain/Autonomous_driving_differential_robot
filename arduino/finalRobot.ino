// Ultrasonic sensor pins
//const int trigPin = 11;
//const int echoPin = 3;

// Motor direction control pins
const int IN1 = 4;  // Left motor forward
const int IN2 = 5;  // Left motor backward
const int IN3 = 6;  // Right motor forward
const int IN4 = 7;  // Right motor backward

// Motor speed control pins (PWM)
const int ENA = 9;  // Left motor speed
const int ENB = 10; // Right motor speed

String inputString = "";
boolean inputComplete = false;

void setup() {
  // Motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Ultrasonic sensor pins
  //pinMode(trigPin, OUTPUT);
  //pinMode(echoPin, INPUT);

  stopLeftMotor();
  stopRightMotor();

  Serial.begin(115200);
  while (!Serial); // Wait for serial
  // Serial.println("Motor Control with Ultrasonic Sensor Ready");
}

void loop() {
  checkSerialCommand();
}

void checkSerialCommand() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      inputComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (inputComplete && inputString.length() >= 1) {
    Serial.print("Received command: ");
    Serial.println(inputString);

    char motor = toLowerCase(inputString.charAt(0));

    // Handle distance request
    //if (inputString.equalsIgnoreCase("d")) {
    //  int distance = readDistance();
    //  Serial.print("Distance: ");
    //  Serial.print(distance);
    //  Serial.println(" cm");
    //}
    // Handle stop command (e.g., "rs", "ls")
    if (inputString.length() == 2 && toLowerCase(inputString.charAt(1)) == 's') {
      if (motor == 'r') {
        stopRightMotor();
        Serial.println("Right motor stopped");
      } else if (motor == 'l') {
        stopLeftMotor();
        Serial.println("Left motor stopped");
      }
    }
    // Handle speed command (e.g., "r200", "l-150")
    else if (inputString.length() > 2) {
      int speed = inputString.substring(1).toInt();
      speed = constrain(speed, -255, 255);

      if (motor == 'r') {
        setRightMotor(speed);
        Serial.print("Right motor set to: ");
        Serial.println(speed);
      } else if (motor == 'l') {
        setLeftMotor(speed);
        Serial.print("Left motor set to: ");
        Serial.println(speed);
      }
    }

    inputString = "";
    inputComplete = false;
  }
}

//int readDistance() {
//  digitalWrite(trigPin, LOW);
//  delayMicroseconds(2);
//  digitalWrite(trigPin, HIGH);
//  delayMicroseconds(10);
//  digitalWrite(trigPin, LOW);
//  long duration = pulseIn(echoPin, HIGH);
//  return duration * 0.034 / 2;
//}

void setRightMotor(int spd) {
  if (spd > 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    analogWrite(ENB, spd);
  } else if (spd < 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    analogWrite(ENB, -spd);
  } else {
    stopRightMotor();
  }
}

void setLeftMotor(int spd) {
  if (spd > 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, spd);
  } else if (spd < 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, -spd);
  } else {
    stopLeftMotor();
  }
}

void stopRightMotor() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);
}

void stopLeftMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
}

char toLowerCase(char c) {
  if (c >= 'A' && c <= 'Z') {
    return c + ('a' - 'A');
  }
  return c;
}
