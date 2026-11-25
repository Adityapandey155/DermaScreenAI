import type { ConditionId } from "./types";

export interface ConditionInfo {
  id: ConditionId;
  name: string;
  description: string;
  guidance: string;
}

// Short educational text only. No treatments or medicines are suggested on purpose.
export const CONDITIONS: Record<ConditionId, ConditionInfo> = {
  eczema_dermatitis: {
    id: "eczema_dermatitis",
    name: "Eczema / Dermatitis",
    description:
      "Eczema and dermatitis are a group of inflammatory skin conditions. They may cause itching, dryness, redness and irritation, and can be triggered by allergies, irritants or other factors.",
    guidance:
      "Many skin conditions look similar to eczema. A healthcare professional can look at the pattern, history and triggers and advise what is appropriate for you.",
  },
  psoriasis: {
    id: "psoriasis",
    name: "Psoriasis",
    description:
      "Psoriasis is a long-lasting condition in which skin cells build up quickly, often forming raised, red or silvery scaly patches, commonly on elbows, knees and scalp.",
    guidance:
      "Psoriasis needs a proper clinical assessment, because it can be confused with eczema or fungal infections. Please see a doctor or dermatologist.",
  },
  fungal_infection: {
    id: "fungal_infection",
    name: "Fungal skin infection",
    description:
      "Fungal skin infections (for example ringworm or tinea) are caused by fungi and often appear as itchy, red, scaly or ring-shaped patches. They are common in warm, humid conditions.",
    guidance:
      "Because many rashes resemble fungal infections, a healthcare professional should confirm what is going on before you decide on any care.",
  },
  urticaria: {
    id: "urticaria",
    name: "Urticaria (hives)",
    description:
      "Urticaria, also called hives, causes raised, itchy welts that can appear and fade within hours. It can be related to allergies, infections or other triggers.",
    guidance:
      "If hives come with swelling of the lips, face or tongue, difficulty breathing or dizziness, seek emergency medical care immediately.",
  },
  insect_bite: {
    id: "insect_bite",
    name: "Insect bite reaction",
    description:
      "Insect bites can cause small, raised, itchy or sore bumps, sometimes with redness or swelling around them. Reactions vary from person to person.",
    guidance:
      "If the area becomes increasingly painful, spreads, forms pus, or you develop fever, please consult a healthcare professional promptly.",
  },
  acne_folliculitis: {
    id: "acne_folliculitis",
    name: "Acne / Folliculitis",
    description:
      "Acne and folliculitis involve inflamed or blocked hair follicles, which may appear as pimples, red bumps or small pus-filled spots, often on the face, back or areas of friction.",
    guidance:
      "If the spots are painful, widespread, scarring or not settling, a healthcare professional can assess the cause and advise on next steps.",
  },
  other: {
    id: "other",
    name: "Other / unclassified condition",
    description:
      "The model grouped this image with skin conditions outside its six main categories. This group contains many different conditions, so it carries very little specific information.",
    guidance:
      "This result does not say that the skin is healthy or unhealthy. Please consult a qualified healthcare professional for a proper evaluation.",
  },
};

export const URGENT_NOTE =
  "Seek medical care promptly if you have a rapidly spreading rash, fever, severe pain, blistering, swelling of the face or lips, difficulty breathing, or a sore or mole that is changing, bleeding or not healing.";

export const DISCLAIMER =
  "This application is an academic AI-based skin screening prototype. Its predictions are intended for educational and informational purposes only and do not constitute medical diagnoses. It cannot replace examination by a qualified dermatologist. Please consult a healthcare professional for proper evaluation.";

export const UNCERTAIN_MESSAGE =
  "Unable to assess this image reliably. Please consult a qualified healthcare professional.";
