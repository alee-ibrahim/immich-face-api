<?php
// Laravel (server side) example. config/services.php:
//   'face_api' => ['url' => env('FACE_API_URL'), 'key' => env('FACE_API_KEY'), 'headers' => []],
// Call it from a queued job, not from a page request (one photo takes about 1-3 s).

namespace App\Services;

use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\Http;

class FaceApiClient
{
    /** @return array{faces: array<int, array>, image: array{w:int,h:int}} */
    public function identify(string $jpegBytes, int $candidates = 3): array
    {
        $response = Http::withToken(config('services.face_api.key'))
            ->withHeaders(config('services.face_api.headers', []))   // e.g. an access-proxy service token
            ->timeout(120)
            ->retry(3, 5000, fn ($e) => $e instanceof RequestException
                && in_array($e->response->status(), [429, 502, 503], true), throw: false)
            ->attach('image', $jpegBytes, 'photo.jpg')
            ->post(rtrim(config('services.face_api.url'), '/').'/v1/identify?candidates='.$candidates);

        $response->throw();   // 400/401/413/415 are bugs or bad input: log and move on

        return $response->json();
    }
}

// Usage:
//   $result = app(FaceApiClient::class)->identify(Storage::get($photo->large_path));
//   foreach ($result['faces'] as $face) {
//       // $face['box'] (fractions), $face['person_id'], $face['name'] (null = unknown),
//       // $face['ambiguous'], $face['candidates']  -> store as pending suggestions for staff to confirm
//   }
